"""DiGress wrapper: shapes and masks, exactness test (a) and (c) of docs/PLAN.md, sampling distribution, determinism.

Runs on a small random-weight DiGress (``digress_fixture``), so it checks mechanics, not generative quality. Skipped
when the DiGress dependencies are not installed (the plain CPU CI job). Test (b) (validity and uniqueness against the
official sampler on a trained model) is ``scripts/verify_digress_wrapper.py``.
"""

from __future__ import annotations

import pytest
import torch

for _mod in ("pytorch_lightning", "torch_geometric", "omegaconf", "wandb", "torchmetrics"):
    pytest.importorskip(_mod)

from vgmg.models.digress import GraphBatch, digress_validity, sample_categorical  # noqa: E402
from vgmg.models.digress_checks import (  # noqa: E402
    check_nfe_accounting, check_posterior_exactness, compare_with_official_sampler)
from vgmg.tests.digress_fixture import build_random_digress, pool_smiles  # noqa: E402
from vgmg.utils.rdkit_utils import canonicalize  # noqa: E402

pytestmark = pytest.mark.digress


@pytest.fixture(scope="module")
def den():
    return build_random_digress()


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def assert_well_formed(state: GraphBatch, den) -> None:
    """One-hot, masked exactly like DiGress's PlaceHolder.mask, symmetric, diagonal = class 0 on real nodes."""
    m = state.node_mask
    assert torch.equal(state.X.sum(-1), m.float())                      # one-hot on real nodes, zero on padding
    pair = m.unsqueeze(1) & m.unsqueeze(2)
    assert torch.equal(state.E.sum(-1), pair.float())
    assert torch.equal(state.E, state.E.transpose(1, 2))
    diag = state.E.diagonal(dim1=1, dim2=2)                             # (B, de, n)
    assert (diag[:, 0, :][m] == 1).all()                                # no self-bonds


def test_sample_categorical_inverse_cdf() -> None:
    probs = torch.tensor([0.2, 0.0, 0.5, 0.3])
    u = torch.tensor([0.0, 0.19, 0.2, 0.69, 0.7, 0.999])
    assert sample_categorical(probs, u).tolist() == [0, 0, 2, 2, 3, 3]   # class 1 has probability 0 and is never drawn


def test_init_shapes_and_masks(den) -> None:
    s = den.init(64, gen())
    assert (s.X.shape, s.E.shape) == ((64, 9, 4), (64, 9, 9, 5))
    assert_well_formed(s, den)
    sizes = s.node_mask.sum(1)
    assert sizes.min() >= 1 and sizes.max() <= 9
    assert (sizes == 9).float().mean() > 0.6                            # QM9 no-H: 83 % of graphs have 9 atoms


def test_propose_groups_children_by_parent(den) -> None:
    parents = den.init(6, gen())
    kids = den.propose(parents, den.num_steps, 4, gen(1))
    assert len(kids) == 24
    assert_well_formed(kids, den)
    assert torch.equal(kids.node_mask, parents.node_mask.repeat_interleave(4, dim=0))   # row i*m+j is child j of parent i
    picked = den.take(kids, torch.arange(6) * 4 + 1)
    assert torch.equal(picked.X, kids.X[1::4]) and len(picked) == 6


def test_exactness_a_posterior_equals_digress(den) -> None:
    """(a) our p(G_{t-1}|G_t) equals the tensors the repository's sample_p_zs_given_zt builds from the same input."""
    r = check_posterior_exactness(den, n=8, times=[den.num_steps, 11, 1], seed=3)
    assert r["max_abs_diff_X"] <= 1e-6 and r["max_abs_diff_E"] <= 1e-6, r


def test_exactness_c_nfe_equals_network_calls() -> None:
    """(c) ``nfe`` equals the number of graphs the network actually processed, including chunked forward passes."""
    r = check_nfe_accounting(build_random_digress(), n=7, m=6, forward_batch=5)
    assert r["nfe_propose"] == 7                     # one pass per parent, independent of m = 6
    assert r["nfe_total"] == 7 + 42 == r["network_graphs"]
    assert r["max_chunk"] <= 5                       # chunking respected


def test_official_sampler_comparison_plumbing(den) -> None:
    """(b) runs end to end and reports sensible numbers (a random-weight model only exercises the plumbing)."""
    r = compare_with_official_sampler(den, n=24, seed=0, batch_size=16)
    assert r["n"] == 24 and set(r["gap"]) == {"validity", "uniqueness"}
    assert r["nfe_per_molecule"] == r["expected_nfe_per_molecule"] == den.num_steps
    assert 0.0 <= r["wrapper_validity_connected"] <= r["wrapper"]["validity"] + 1e-9


def test_children_follow_the_posterior(den) -> None:
    """Empirical child frequencies of one parent match the probabilities (the inverse-CDF sampler is unbiased)."""
    parent = den.init(1, gen(5))
    t, m = 12, 20000
    prob_X, prob_E = den.posterior_probs(parent, t)
    kids = den.propose(parent, t, m, gen(6))
    n = int(parent.node_mask.sum())
    freq_X = kids.X[:, :n].mean(0)                                       # (n, dx)
    assert (freq_X - prob_X[0, :n]).abs().max() < 0.02                   # 5 sigma at m = 20000
    freq_E = kids.E[:, :n, :n].mean(0)
    off = ~torch.eye(n, dtype=torch.bool)
    assert (freq_E[off] - prob_E[0, :n, :n][off]).abs().max() < 0.02


def test_determinism_and_generator_independence(den) -> None:
    a = den.propose(den.init(5, gen(1)), 7, 3, gen(2))
    b = den.propose(den.init(5, gen(1)), 7, 3, gen(2))
    c = den.propose(den.init(5, gen(1)), 7, 3, gen(3))
    assert torch.equal(a.X, b.X) and torch.equal(a.E, b.E)
    assert not torch.equal(a.E, c.E)
    torch.manual_seed(0)                                                  # the global RNG must not matter
    d = den.propose(den.init(5, gen(1)), 7, 3, gen(2))
    assert torch.equal(a.E, d.E)


def test_features_are_normalised_histograms(den) -> None:
    f = den.features(den.encode(["CC=O", "C1CC1", "C#N"]))
    assert f.shape == (3, den.feature_dim) == (3, 9)
    # C C=O -> atoms C2 N0 O1 F0 | single 1 double 1 triple 0 aromatic 0 | n = 3, all divided by 9
    assert torch.allclose(f[0], torch.tensor([2, 0, 1, 0, 1, 1, 0, 0, 3]) / 9)
    assert torch.allclose(f[1], torch.tensor([3, 0, 0, 0, 3, 0, 0, 0, 3]) / 9)
    assert torch.allclose(f[2], torch.tensor([1, 1, 0, 0, 0, 0, 1, 0, 2]) / 9)


def test_encode_decode_round_trip(den) -> None:
    smiles, nfe0 = pool_smiles(), den.nfe
    back = den.decode(den.encode(smiles))
    # an aromatic N-H cannot be written without explicit hydrogens (DiGress's QM9 convention): the only allowed failures
    bad = [a for a, b in zip(smiles, back) if canonicalize(a) != canonicalize(b)]
    assert all("[nH]" in canonicalize(a) for a in bad), bad
    assert len(bad) < len(smiles) // 10
    assert den.nfe == nfe0                                                # encode/decode never touch the network


def test_encode_rejects_unsupported_molecules(den) -> None:
    for smi in ("CCCCCCCCCC", "CCl", "not-a-smiles"):
        with pytest.raises(ValueError):
            den.encode([smi])


def test_predict_x0_counts_one_pass_per_graph_and_returns_strings() -> None:
    den = build_random_digress()
    out = den.predict_x0(den.init(9, gen()), 5)
    assert len(out) == 9 and all(isinstance(s, str) for s in out) and den.nfe == 9


def test_checkpoint_with_metric_buffers_loads(den, tmp_path) -> None:
    """A real Lightning checkpoint carries ``sampling_metrics.*`` buffers (found on the first trained checkpoint)."""
    from vgmg.models.digress import DIGRESS_ROOT, DiGressDenoiser, import_digress, load_digress_checkpoint, network_state

    state = {**den.model.state_dict(), "sampling_metrics.n_target_dist": torch.zeros(3),
             "sampling_metrics.edge_target_dist": torch.zeros(5)}
    assert set(state) - set(network_state(state)) == {"sampling_metrics.n_target_dist", "sampling_metrics.edge_target_dist"}
    hparams = {k: v for k, v in den.model.hparams.items() if k not in ("train_metrics", "sampling_metrics")}
    path = tmp_path / "fake.ckpt"
    torch.save({"hyper_parameters": hparams, "state_dict": state}, path)
    loaded = DiGressDenoiser(path, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(loaded.model.state_dict().values(), den.model.state_dict().values()))
    state["model.unrelated.weight"] = torch.zeros(1)                      # a non-metric extra key must still fail loudly
    torch.save({"hyper_parameters": hparams, "state_dict": state}, path)
    with pytest.raises(RuntimeError):
        load_digress_checkpoint(import_digress(), path)
    assert DIGRESS_ROOT.exists()


def test_digress_validity_convention() -> None:
    assert digress_validity(["CCO", "CC.O", "", "C(C)(C)(C)(C)C"]) == 0.5   # fragments allowed; empty / bad valence not


def test_svdd_runs_through_the_wrapper() -> None:
    """The unchanged samplers drive the wrapper: M = 1, a value-function scorer, and the PM scorer."""
    from vgmg.models.value_function import MLPValueFunction
    from vgmg.rl.rewards import get_reward
    from vgmg.rl.svdd import pm_scorer, svdd_sample, value_scorer

    den = build_random_digress()
    T = den.num_steps
    out = svdd_sample(den, None, 4, 1, gen())
    assert len(out) == 4 and den.nfe == 4 * T
    den.nfe = 0
    vf = MLPValueFunction(den.feature_dim)
    out = svdd_sample(den, value_scorer(vf), 4, 3, gen())
    assert len(out) == 4 and den.nfe == 4 * T                              # SVDD-MLP: T passes per molecule for any M
    den.nfe = 0
    out = svdd_sample(den, pm_scorer(get_reward("sa")), 4, 3, gen())
    assert len(out) == 4 and den.nfe == 4 * (3 + 1) * T                    # SVDD-PM: (M+1)*T passes per molecule


def test_value_diagnostics_on_real_qm9_molecules_and_digress_states() -> None:
    """The t = 0 check encodes real QM9 validation molecules; the sibling diagnostic runs on DiGress states."""
    from vgmg.config import load_config
    from vgmg.experiments import heldout_molecules, value_diagnostics
    from vgmg.models.value_function import MLPValueFunction

    den = build_random_digress()
    cfg = load_config("configs/smoke_test.yaml", ["data.reference=qm9", "svdd.num_candidates_m=3",
                      "value_fn.diag_t0_molecules=40", "value_fn.diag_sibling_parents=4",
                      "value_fn.diag_sibling_completions=2", "value_fn.diag_sibling_t_fracs=[0.5]"])
    mols = heldout_molecules(cfg, den, 40)
    assert len(mols) == 40 and all(canonicalize(m) for m in mols)
    out = value_diagnostics(cfg, den, MLPValueFunction(den.feature_dim, 16, 1), gen())
    assert out["t0_check"]["n"] == 40 and len(out["sibling"]) == 1 and out["sibling"][0]["t"] == den.num_steps // 2
    assert out["diag_nfe"] > 0


def test_gnn_value_function_is_permutation_invariant_and_trains_through_the_pipeline(tmp_path) -> None:
    from vgmg.config import load_config
    from vgmg.experiments import get_value_function
    from vgmg.models.value_function import GNNValueFunction

    den = build_random_digress()
    gnn = GNNValueFunction(*den.graph_shape, hidden_dim=16, num_layers=3)
    state = den.encode(["CC(=O)N", "C1CC1O"])
    flat = den.graph_features(state)
    assert flat.shape == (2, 9 * 4 + 81 * 5 + 9)
    # relabelling the atoms of a graph must not change the prediction (message passing + pooling are symmetric)
    perm = torch.tensor([3, 0, 1, 2, 4, 5, 6, 7, 8])
    p = GraphBatch(state.X[:, perm], state.E[:, perm][:, :, perm], state.node_mask[:, perm])
    t = torch.tensor(0.5)
    assert torch.allclose(gnn(flat, t), gnn(den.graph_features(p), t), atol=1e-5)
    assert (gnn(flat, t) != gnn(den.graph_features(den.encode(["CCCC", "C1CC1O"])), t)).any()  # not constant
    # end to end: rollouts, training, diagnostics and scoring all select the graph featurizer through feature_kind
    cfg = load_config("configs/smoke_test.yaml", [f"output_dir={tmp_path}", "svdd.value_fn_arch=gnn", "svdd.num_candidates_m=3",
                      "value_fn.train_chains=24", "value_fn.val_chains=8", "value_fn.ts_per_chain=2", "value_fn.train_steps=20",
                      "value_fn.eval_every=10", "value_fn.gnn_hidden_dim=16", "value_fn.diag_t0_molecules=8",
                      "value_fn.diag_sibling_parents=2", "value_fn.diag_sibling_completions=1", "value_fn.diag_sibling_t_fracs=[0.5]",
                      "data.reference=qm9"])
    model, diag = get_value_function(cfg, den)
    assert isinstance(model, GNNValueFunction) and diag["t0_check"]["n"] == 8 and diag["n_train_rows"] == 48
