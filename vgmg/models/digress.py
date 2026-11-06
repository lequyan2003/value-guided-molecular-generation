"""DiGress wrapper (inference only). Implements ``vgmg.models.denoiser.Denoiser``.

What this module is for
-----------------------
SVDD needs to stop DiGress at every reverse step, draw ``m`` children x_{t-1} ~ p(. | x_t), score them and keep one.
DiGress's own ``sample_batch`` runs the whole chain internally and draws one child, so the reverse step is re-implemented
here, *adapted from* ``DiscreteDenoisingDiffusion.sample_p_zs_given_zt`` (cvignac/DiGress @ bdfa10c, MIT; declared in
CONTRIBUTIONS.md). Everything that is not the final sampling is DiGress's own code: the network, the extra features
(``compute_extra_data``), the transition matrices and the closed-form posterior
(``diffusion_utils.compute_batched_over0_posterior_distribution``).

One reverse step (what ``propose`` does)
  1. one network forward on the ``n`` parents gives p_hat(G_0 | G_t);
  2. p(G_s | G_t) = sum_{G_0} q(G_s | G_t, G_0) p_hat(G_0 | G_t) per node and per edge (``posterior_probs``);
  3. ``m`` independent samples are drawn from those same probabilities.
Steps 1-2 do not depend on ``m``, so a step costs one forward pass per *parent* for any ``m`` (``nfe``); this is why
SVDD-MLP costs T passes per molecule while best-of-N costs N*T.

State format (``GraphBatch``)
  Dense, padded to ``max_n_nodes`` (9 for QM9 without hydrogens): ``X`` one-hot atom types (B, n, dx), ``E`` one-hot
  bond classes (B, n, n, de) with class 0 = "no bond", symmetric, and ``node_mask`` (B, n). Padded rows are all-zero,
  exactly as in DiGress's ``PlaceHolder.mask``. Children of parent ``i`` sit at rows ``i*m + j``.

Differences from the repository, all deliberate
  * Sampling uses inverse-CDF draws from the *caller's* ``torch.Generator`` (DiGress draws from the global RNG), so runs
    are reproducible per seed and independent of other code. The distribution is the same (test_digress_wrapper.py).
  * Graphs are always padded to ``max_n_nodes`` instead of to the batch maximum. The network and all extra features are
    masked, so this changes nothing but a few wasted multiplications.
  * Validity: ``decode`` keeps disconnected graphs as ``"A.B"`` SMILES; the project's single validity definition
    (``vgmg.utils.rdkit_utils.mol_from_smiles``) rejects them. DiGress's own convention (largest fragment) is
    obtained with ``digress_validity``.

Verified facts (docs/VERIFICATION.md): the official QM9 checkpoint link is dead (404, issue #109), so DiGress is trained by
us with the unmodified ``qm9_no_h`` config and frozen. DiGress was tested on Python 3.9, PyTorch 2.0.1, CUDA 11.8,
torch_geometric 2.3.1, rdkit 2023.03.2; this module therefore avoids 3.10-only syntax.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import RDLogger

from vgmg.constants import REPO_ROOT
from vgmg.utils.rdkit_utils import is_sanitizable

DIGRESS_ROOT = REPO_ROOT / "third_party" / "DiGress"
RDLogger.DisableLog("rdApp.*")  # random / early-training graphs violate valence constantly; failures are counted, not logged
INVALID_SMILES = ""  # what ``decode`` returns for a graph RDKit cannot sanitize; fails every validity check
N_FEATURES_EXTRA = 1  # node count, on top of the atom and bond histograms


def import_digress() -> Any:
    """Make the pinned DiGress sources importable (they use both ``src.*`` and top-level ``models``/``diffusion``)."""
    for p in (DIGRESS_ROOT, DIGRESS_ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import diffusion_model_discrete  # noqa: WPS433  (needs the DiGress environment; see scripts/setup_digress_env.sh)
    return diffusion_model_discrete


def load_digress_checkpoint(mod: Any, checkpoint: str | Path) -> Any:
    """Rebuild a ``DiscreteDenoisingDiffusion`` from a DiGress Lightning checkpoint, on CPU.

    The constructor arguments (config, dataset statistics, feature modules) are stored in the checkpoint's
    ``hyper_parameters``; the metric objects are not and are not needed for inference. Done by hand instead of
    ``load_from_checkpoint`` because PyTorch >= 2.6 defaults ``torch.load`` to ``weights_only=True``, which cannot
    unpickle those objects; the checkpoint is our own training output, so full unpickling is safe.
    """
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    kwargs = dict(ckpt["hyper_parameters"], train_metrics=None, sampling_metrics=None)
    model = mod.DiscreteDenoisingDiffusion(**kwargs)
    model.load_state_dict(network_state(ckpt["state_dict"]))     # strict: any other mismatch is a real error
    return model


METRIC_PREFIXES = ("sampling_metrics.", "train_metrics.", "val_metrics.", "test_metrics.")


def network_state(state: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint's state dict without DiGress's metric objects.

    Lightning checkpoints hold the buffers of the training-time metric modules (for example the target
    node/edge/valency distributions of ``sampling_metrics``). They are not network weights and the wrapper builds the
    model without metric objects, so these keys are dropped; every other key must still match exactly.
    """
    return {k: v for k, v in state.items() if not k.startswith(METRIC_PREFIXES)}


@dataclass
class GraphBatch:
    """A batch of dense, padded, one-hot graphs (see module docstring)."""

    X: torch.Tensor          # (B, n, dx) float
    E: torch.Tensor          # (B, n, n, de) float
    node_mask: torch.Tensor  # (B, n) bool

    def __len__(self) -> int:
        return self.X.shape[0]

    def take(self, idx: torch.Tensor) -> "GraphBatch":
        """Sub-batch at row indices ``idx``."""
        idx = idx.to(self.X.device)
        return GraphBatch(self.X[idx], self.E[idx], self.node_mask[idx])

    def to(self, device: torch.device | str) -> "GraphBatch":
        """Move to ``device``."""
        return GraphBatch(self.X.to(device), self.E.to(device), self.node_mask.to(device))

    def chunks(self, size: int) -> list["GraphBatch"]:
        """Consecutive sub-batches of at most ``size`` graphs (forward-pass memory control)."""
        return [GraphBatch(self.X[i:i + size], self.E[i:i + size], self.node_mask[i:i + size])
                for i in range(0, len(self), size)]

    @staticmethod
    def cat(parts: Sequence["GraphBatch"]) -> "GraphBatch":
        """Concatenate batches along the batch dimension."""
        return GraphBatch(torch.cat([p.X for p in parts]), torch.cat([p.E for p in parts]),
                          torch.cat([p.node_mask for p in parts]))


def sample_categorical(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF draw: index of the first class whose cumulative probability reaches ``u`` in [0, 1).

    ``probs`` (..., d) broadcasts against ``u`` (...) (extra leading dims of ``u`` are fine). Deterministic given ``u``.
    """
    cdf = probs.cumsum(-1)
    return (u.unsqueeze(-1) >= cdf).sum(-1).clamp_(max=probs.shape[-1] - 1)


class DiGressDenoiser:
    """A frozen DiGress QM9 model behind the ``Denoiser`` interface.

    ``checkpoint`` is a Lightning checkpoint written by DiGress training (``last.ckpt`` of the run). The model's
    hyper-parameters (config, dataset statistics, feature modules) are stored inside it, so no dataset is needed.
    ``model`` lets tests pass an already built ``DiscreteDenoisingDiffusion`` instead.
    """

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cuda", model: Any = None,
                 max_forward_batch: int = 2048) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        mod = import_digress()
        if model is None:
            model = load_digress_checkpoint(mod, checkpoint)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.info = self.model.dataset_info
        self.num_steps: int = int(self.model.T)
        self.max_n: int = int(self.info.max_n_nodes)
        self.dx: int = int(self.model.Xdim_output)
        self.de: int = int(self.model.Edim_output)
        self.feature_dim: int = self.dx + (self.de - 1) + N_FEATURES_EXTRA
        self.max_forward_batch = max_forward_batch
        self.nfe = 0
        # DiGress sampling code, imported after ``import_digress`` has put the pinned sources on sys.path
        from src.diffusion import diffusion_utils  # noqa: WPS433
        self._du = diffusion_utils

    # ------------------------------------------------------------------ sampling primitives
    def _uniform(self, shape: Sequence[int], generator: torch.Generator) -> torch.Tensor:
        """U[0,1) draws from the caller's generator, moved to the model device (CPU generators work with CUDA models)."""
        return torch.rand(tuple(shape), generator=generator, device=generator.device).to(self.device)

    def _sample_graphs(self, prob_X: torch.Tensor, prob_E: torch.Tensor, node_mask: torch.Tensor, m: int,
                       generator: torch.Generator) -> GraphBatch:
        """``m`` i.i.d. graphs per row from per-node / per-edge categorical probabilities (adapted from
        ``diffusion_utils.sample_discrete_features``): edges are drawn for the upper triangle and mirrored; padded
        rows and the diagonal follow DiGress (diagonal = class 0 on real nodes, padding = all-zero)."""
        b, n, _ = prob_X.shape
        x = sample_categorical(prob_X.unsqueeze(1), self._uniform((b, m, n), generator))        # (b, m, n)
        e = sample_categorical(prob_E.unsqueeze(1), self._uniform((b, m, n, n), generator))     # (b, m, n, n)
        e = torch.triu(e, diagonal=1)
        e = e + e.transpose(-1, -2)
        x, e = x.reshape(b * m, n), e.reshape(b * m, n, n)
        mask = node_mask.repeat_interleave(m, dim=0)
        X = F.one_hot(x, self.dx).float() * mask.unsqueeze(-1)
        E = F.one_hot(e, self.de).float() * (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
        return GraphBatch(X, E, mask)

    def init(self, n: int, generator: torch.Generator) -> GraphBatch:
        """``n`` states x_T: node counts from the training distribution, types i.i.d. from the training marginals
        (``transition: marginal`` makes these the limit distribution of the forward process)."""
        counts = sample_categorical(self.model.node_dist.prob.to(self.device), self._uniform((n,), generator))
        arange = torch.arange(self.max_n, device=self.device).unsqueeze(0)
        mask = arange < counts.unsqueeze(1)
        limit = self.model.limit_dist
        prob_X = limit.X.to(self.device).expand(n, self.max_n, self.dx)
        prob_E = limit.E.to(self.device).expand(n, self.max_n, self.max_n, self.de)
        return self._sample_graphs(prob_X, prob_E, mask, 1, generator)

    # ------------------------------------------------------------------ network and posterior
    def _network_probs(self, state: GraphBatch, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax of the network's clean-graph prediction p_hat(G_0 | G_t) for states at integer time ``t``.

        Counts one forward pass per graph in ``nfe``. Large batches are split to bound memory.
        """
        out_X, out_E = [], []
        for part in state.chunks(self.max_forward_batch):
            b = len(part)
            t_norm = torch.full((b, 1), t / self.num_steps, device=self.device)
            noisy = {"X_t": part.X, "E_t": part.E, "y_t": part.X.new_zeros((b, 0)), "t": t_norm,
                     "node_mask": part.node_mask}
            extra = self.model.compute_extra_data(noisy)
            pred = self.model.forward(noisy, extra, part.node_mask)
            out_X.append(F.softmax(pred.X, dim=-1))
            out_E.append(F.softmax(pred.E, dim=-1))
            self.nfe += b
        return torch.cat(out_X), torch.cat(out_E)

    @torch.no_grad()
    def posterior_probs(self, state: GraphBatch, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """p(G_{t-1} | G_t) as per-node ``(B, n, dx)`` and per-edge ``(B, n, n, de)`` probabilities.

        Same computation as the first half of ``sample_p_zs_given_zt`` (reverse step t -> s = t-1). Test (a) of
        docs/PLAN.md compares the result with the repository's own tensors on the same input.
        """
        model, du = self.model, self._du
        b, n, _ = state.X.shape
        ones = torch.ones((b, 1), device=self.device)
        s_norm, t_norm = ones * (t - 1) / self.num_steps, ones * t / self.num_steps
        beta_t = model.noise_schedule(t_normalized=t_norm)
        alpha_s_bar = model.noise_schedule.get_alpha_bar(t_normalized=s_norm)
        alpha_t_bar = model.noise_schedule.get_alpha_bar(t_normalized=t_norm)
        Qtb = model.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = model.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = model.transition_model.get_Qt(beta_t, self.device)

        pred_X, pred_E = self._network_probs(state, t)
        post_X = du.compute_batched_over0_posterior_distribution(X_t=state.X, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X)
        post_E = du.compute_batched_over0_posterior_distribution(X_t=state.E, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)
        un_X = (pred_X.unsqueeze(-1) * post_X).sum(dim=2)                                    # (b, n, d_{t-1})
        un_X[un_X.sum(dim=-1) == 0] = 1e-5
        prob_X = un_X / un_X.sum(dim=-1, keepdim=True)
        pred_E = pred_E.reshape(b, -1, pred_E.shape[-1])
        un_E = (pred_E.unsqueeze(-1) * post_E).sum(dim=-2)
        un_E[un_E.sum(dim=-1) == 0] = 1e-5
        prob_E = (un_E / un_E.sum(dim=-1, keepdim=True)).reshape(b, n, n, -1)
        return prob_X, prob_E

    # ------------------------------------------------------------------ Denoiser interface
    @torch.no_grad()
    def propose(self, state: GraphBatch, t: int, m: int, generator: torch.Generator) -> GraphBatch:
        """``m`` i.i.d. children x_{t-1} ~ p(. | x_t) per parent; one forward pass per parent, whatever ``m``."""
        prob_X, prob_E = self.posterior_probs(state, t)
        return self._sample_graphs(prob_X, prob_E, state.node_mask, m, generator)

    def take(self, state: GraphBatch, idx: torch.Tensor) -> GraphBatch:
        """Sub-batch at row indices ``idx``."""
        return state.take(idx)

    @torch.no_grad()
    def features(self, state: GraphBatch) -> torch.Tensor:
        """Value-function input ``(B, dx + de - 1 + 1)`` on CPU: atom-type counts, bond-type counts (single, double,
        triple, aromatic; each bond counted once) and the node count, all divided by ``max_n_nodes``.

        A lossy summary on purpose (proposal: "atom/bond histograms"): it cannot see which atoms are bonded, hence
        no rings or fragments. Defined for noisy states too: the histograms are those of the current noisy graph.
        """
        atoms = state.X.sum(dim=1)                                  # (B, dx); padded rows are zero
        bonds = state.E[..., 1:].sum(dim=(1, 2)) / 2                # (B, de-1); E is symmetric
        n_atoms = state.node_mask.sum(dim=1, keepdim=True).float()
        return (torch.cat([atoms, bonds, n_atoms], dim=1) / self.max_n).cpu()

    @property
    def graph_shape(self) -> tuple[int, int, int]:
        """``(max_nodes, atom_classes, bond_classes)``: how ``graph_features`` is laid out (for the GNN value function)."""
        return self.max_n, self.dx, self.de

    @torch.no_grad()
    def graph_features(self, state: GraphBatch) -> torch.Tensor:
        """Flat dense graph ``(B, n*dx + n*n*de + n)`` on CPU: atom one-hots, bond one-hots, node mask (GNN input)."""
        b = len(state)
        return torch.cat([state.X.reshape(b, -1), state.E.reshape(b, -1), state.node_mask.float()], dim=1).cpu()

    @staticmethod
    def _collapse(state: GraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Argmax atom types ``(B, n)`` and bond classes ``(B, n, n)`` of a one-hot batch."""
        return state.X.argmax(dim=-1), state.E.argmax(dim=-1)

    def to_graphs(self, state: GraphBatch) -> list[list[torch.Tensor]]:
        """DiGress's molecule-list format ``[atom_types (n,), edge_types (n, n)]`` per graph, on CPU (padding cut)."""
        X, E = self._collapse(state)
        out = []
        for i in range(len(state)):
            n = int(state.node_mask[i].sum())
            out.append([X[i, :n].cpu(), E[i, :n, :n].cpu()])
        return out

    def decode(self, state: GraphBatch) -> list[str]:
        """SMILES of each graph via DiGress's ``build_molecule`` + ``mol2smiles`` (``INVALID_SMILES`` on failure).

        A disconnected graph yields ``"A.B"``: parsable (valid under DiGress's largest-fragment convention) but
        rejected by this project's connected-only validity.
        """
        from analysis.rdkit_functions import build_molecule, mol2smiles  # noqa: WPS433  (DiGress's own decoder)
        out = []
        for atom_types, edge_types in self.to_graphs(state):
            smiles = mol2smiles(build_molecule(atom_types, edge_types, self.info.atom_decoder))
            out.append(smiles if smiles else INVALID_SMILES)
        return out

    @torch.no_grad()
    def predict_x0(self, state: GraphBatch, t: int) -> list[str]:
        """Decode argmax of the network's own clean-graph prediction for states at time ``t`` (one pass per graph)."""
        pred_X, pred_E = self._network_probs(state, t)
        mask = state.node_mask
        X = F.one_hot(pred_X.argmax(-1), self.dx).float() * mask.unsqueeze(-1)
        e = pred_E.argmax(-1)
        e = torch.triu(e, diagonal=1)
        e = e + e.transpose(-1, -2)
        E = F.one_hot(e, self.de).float() * (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
        return self.decode(GraphBatch(X, E, mask))

    def encode(self, smiles: Sequence[str]) -> GraphBatch:
        """Clean graphs (t = 0) of real molecules; inverse of ``decode`` up to atom order and kekulization.

        Uses the QM9 conventions of DiGress: heavy atoms only, bond classes 1-4 = single, double, triple, aromatic.
        Raises ``ValueError`` for molecules with unsupported atoms or more than ``max_n_nodes`` heavy atoms.
        """
        from rdkit import Chem  # noqa: WPS433
        bond_class = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
                      Chem.BondType.AROMATIC: 4}
        enc = {a: i for i, a in enumerate(self.info.atom_decoder)}
        b, n = len(smiles), self.max_n
        x = torch.zeros(b, n, dtype=torch.long)
        e = torch.zeros(b, n, n, dtype=torch.long)
        mask = torch.zeros(b, n, dtype=torch.bool)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() > n:
                raise ValueError(f"cannot encode {smi!r}: unparsable or more than {n} heavy atoms")
            for a in mol.GetAtoms():
                if a.GetSymbol() not in enc:
                    raise ValueError(f"cannot encode {smi!r}: atom {a.GetSymbol()} not in {list(enc)}")
                x[i, a.GetIdx()] = enc[a.GetSymbol()]
            mask[i, :mol.GetNumAtoms()] = True
            for bd in mol.GetBonds():
                u, v = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
                e[i, u, v] = e[i, v, u] = bond_class[bd.GetBondType()]
        m = mask.to(self.device)
        X = F.one_hot(x.to(self.device), self.dx).float() * m.unsqueeze(-1)
        E = F.one_hot(e.to(self.device), self.de).float() * (m.unsqueeze(1) & m.unsqueeze(2)).unsqueeze(-1)
        return GraphBatch(X, E, m)


def digress_validity(smiles: Sequence[str]) -> float:
    """Fraction of ``decode`` outputs that are valid under DiGress's convention (sanitizable, fragments allowed).

    This project's primary validity additionally requires one connected fragment
    (``vgmg.eval.metrics.compute_metrics``); both are reported so the numbers can be compared with the DiGress paper.
    """
    return sum(is_sanitizable(s) for s in smiles) / max(len(smiles), 1)


__all__ = ["DiGressDenoiser", "GraphBatch", "INVALID_SMILES", "digress_validity", "import_digress", "load_digress_checkpoint",
           "sample_categorical"]
