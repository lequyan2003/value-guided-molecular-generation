"""A small randomly initialised DiGress (qm9_no_h architecture, fewer/narrower layers) for wrapper tests.

The weights are random, so samples are not molecules of any quality: these fixtures test *mechanics* (shapes, masks,
probabilities, pass counting, reproducibility), never generative quality. Needs the DiGress dependencies
(pytorch_lightning, torch_geometric, omegaconf, wandb, torchmetrics); tests skip when they are missing.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from vgmg.constants import SMOKE_SMILES
from vgmg.models.digress import DIGRESS_ROOT, DiGressDenoiser, import_digress


def build_random_digress(n_layers: int = 2, steps: int = 20, seed: int = 0) -> DiGressDenoiser:
    """Random-weight ``DiscreteDenoisingDiffusion`` with the qm9_no_h dataset statistics, wrapped for inference."""
    import_digress()
    from datasets.qm9_dataset import QM9infos
    from diffusion.extra_features import ExtraFeatures
    from diffusion.extra_features_molecular import ExtraMolecularFeatures
    from diffusion_model_discrete import DiscreteDenoisingDiffusion
    from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
    from torch_geometric.data import Batch, Data

    conf = DIGRESS_ROOT / "configs"
    cfg = OmegaConf.create({"general": OmegaConf.load(conf / "general/general_default.yaml"),
                            "model": OmegaConf.load(conf / "model/discrete.yaml"),
                            "train": OmegaConf.load(conf / "train/train_default.yaml"),
                            "dataset": OmegaConf.load(conf / "dataset/qm9.yaml")})
    cfg.model.n_layers, cfg.model.diffusion_steps = n_layers, steps
    cfg.model.hidden_mlp_dims = {"X": 32, "E": 16, "y": 16}
    cfg.model.hidden_dims = {"dx": 32, "de": 16, "dy": 16, "n_head": 4, "dim_ffX": 32, "dim_ffE": 16, "dim_ffy": 16}

    infos = QM9infos(datamodule=None, cfg=cfg)
    extra, domain = ExtraFeatures(cfg.model.extra_features, dataset_info=infos), ExtraMolecularFeatures(infos)

    class _OneBatch:  # stands in for the datamodule: ``compute_input_output_dims`` only reads one batch
        @staticmethod
        def train_dataloader() -> list:
            graphs = []
            for smi in ("CCO", "C1CC1"):
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smi)
                ei = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
                ei = torch.tensor(ei + [[j, i] for i, j in ei]).t().contiguous()
                x = torch.nn.functional.one_hot(torch.zeros(mol.GetNumAtoms(), dtype=torch.long), 4).float()
                ea = torch.nn.functional.one_hot(torch.ones(ei.shape[1], dtype=torch.long), 5).float()
                graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, y=torch.zeros(1, 0)))
            return [Batch.from_data_list(graphs)]

    infos.compute_input_output_dims(datamodule=_OneBatch(), extra_features=extra, domain_features=domain)
    torch.manual_seed(seed)
    model = DiscreteDenoisingDiffusion(cfg=cfg, dataset_infos=infos, train_metrics=TrainAbstractMetricsDiscrete(),
                                       sampling_metrics=None, visualization_tools=None, extra_features=extra,
                                       domain_features=domain)
    return DiGressDenoiser(model=model, device="cpu")


def pool_smiles() -> list[str]:
    """Heavy-atom-only molecules of at most 9 atoms made of C/N/O/F (valid inputs for ``encode``)."""
    return [s for s in SMOKE_SMILES if all(ch not in s for ch in ("Cl", "Br")) and sum(c.isalpha() for c in s) <= 9]
