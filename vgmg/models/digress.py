"""DiGress wrapper (inference only). Implements ``vgmg.models.denoiser.Denoiser``.

TODO(Stage 2, re-dated 2026-11-05: write Nov 6-7, exactness tests before the Sun Nov 8 12:00 freeze; see docs/PLAN.md).
Verified facts (docs/VERIFICATION.md):
  1. Pin cvignac/DiGress @ bdfa10c (git submodule). The official QM9 checkpoint link is DEAD
     (404, issue #109): we train DiGress on QM9 with the unmodified ``qm9_no_h`` config and
     freeze it. Gate G2: DiGress-metric validity >= 97% (paper: 99.0 valid, 96.2 unique).
  2. ``sample_p_zs_given_zt`` does ONE network forward per call and then samples
     x_{t-1} from sum_x0 p(x0|x_t) q(x_{t-1}|x_t,x0); ``propose`` re-implements that posterior
     (adapted from the repo, MIT; declare in CONTRIBUTIONS.md) and draws m samples from the
     same probabilities, so it costs 1 NFE per parent for any m. Exactness test: our
     prob_X/prob_E must equal the repo's on the same noisy input.
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
