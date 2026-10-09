"""DiGress wrapper (inference only). Implements ``vgmg.models.denoiser.Denoiser``.

TODO(week 2, after GAIVI gate G1). Facts to verify when writing it, in this order:
  1. Pin cvignac/DiGress @ bdfa10c; the QM9 (heavy-atom) checkpoint must load and
     reproduce the published validity (gate G2). The README says the author lost access
     to some checkpoints: confirm the QM9 download link still works before anything else.
  2. Read ``sample_p_zs_given_zt``: it should do ONE network forward per parent and then
     sample x_{t-1} from q(x_{t-1}|x_t,x0) p(x0|x_t); ``propose`` then costs 1 NFE per parent
     for any m (this is the basis of the NFE accounting in ``denoiser.py``).
  3. DiGress's own extra features (cycles, spectral) are recomputed from the noisy graph at
     every step, so ``propose`` must rebuild them for the parents only.
  4. ``features`` = atom-type histogram + bond-type histogram + node count (fixed width).
  5. Decoding: build an RDKit mol from argmax atom/bond types; DiGress's own validity
     keeps the largest fragment, ours does not (disconnected = invalid, see rdkit_utils).
Environment note: DiGress was tested on Python 3.9, PyTorch 2.0.1, CUDA 11.8,
torch_geometric 2.3.1, rdkit 2023.03.2. This package therefore avoids 3.10-only syntax
(``from __future__ import annotations`` everywhere) so it runs in that environment.
"""

from __future__ import annotations


class DiGressDenoiser:
    """Pretrained DiGress QM9 model behind the ``Denoiser`` interface."""

    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        raise NotImplementedError("DiGress wrapper is not implemented yet (week 2, see module docstring)")
