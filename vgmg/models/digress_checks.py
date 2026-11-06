"""Exactness checks for ``DiGressDenoiser`` (docs/PLAN.md, Stage 2, tests (a)-(c)).

Each function returns plain numbers so the same code serves the unit tests (random-weight model, mechanics only) and
``scripts/verify_digress_wrapper.py`` (trained checkpoint, the gate that must pass before any result is produced).

(a) ``check_posterior_exactness``: our p(G_{t-1} | G_t) equals the tensors DiGress's own ``sample_p_zs_given_zt`` builds
    from the same noisy input.
(b) ``compare_with_official_sampler``: unguided samples (M = 1) have the same validity and uniqueness, under DiGress's own
    metric code, as samples from DiGress's ``sample_batch``.
(c) ``check_nfe_accounting``: ``nfe`` equals the number of graphs the network actually processed (counted with a hook).
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from vgmg.eval.metrics import compute_metrics
from vgmg.models.digress import DiGressDenoiser
from vgmg.rl.rewards import get_reward
from vgmg.rl.svdd import svdd_sample


def check_posterior_exactness(den: DiGressDenoiser, n: int = 16, times: Sequence[int] | None = None,
                              seed: int = 0) -> dict[str, float]:
    """(a) Largest absolute difference between our posterior probabilities and the repository's, over ``times``.

    The repository's tensors are captured by wrapping ``diffusion_utils.sample_discrete_features``, which receives
    exactly ``prob_X`` / ``prob_E`` inside ``sample_p_zs_given_zt``. Inputs are fresh x_T states pushed through a few
    of our own reverse steps so that mid-chain (not only pure-noise) states are covered.
    """
    from src.diffusion import diffusion_utils  # noqa: WPS433

    T = den.num_steps
    times = list(times) if times is not None else [T, max(T * 3 // 4, 1), max(T // 2, 1), max(T // 10, 1), 1]
    gen = torch.Generator().manual_seed(seed)
    state, t_now = den.init(n, gen), T
    worst = {"X": 0.0, "E": 0.0}
    real = diffusion_utils.sample_discrete_features
    for t in sorted(times, reverse=True):
        while t_now > t:                                         # walk the chain down to time t with our own sampler
            state = den.propose(state, t_now, 1, gen)
            t_now -= 1
        seen: dict[str, torch.Tensor] = {}

        def spy(probX: torch.Tensor, probE: torch.Tensor, node_mask: torch.Tensor) -> Any:
            seen["X"], seen["E"] = probX.clone(), probE.clone()  # the repository mutates both in place afterwards
            return real(probX, probE, node_mask)

        diffusion_utils.sample_discrete_features = spy
        try:
            ones = torch.ones((n, 1), device=den.device)
            den.model.sample_p_zs_given_zt(ones * (t - 1) / T, ones * t / T, state.X, state.E,
                                           state.X.new_zeros((n, 0)), state.node_mask)
        finally:
            diffusion_utils.sample_discrete_features = real
        ours_X, ours_E = den.posterior_probs(state, t)
        worst["X"] = max(worst["X"], float((ours_X - seen["X"]).abs().max()))
        worst["E"] = max(worst["E"], float((ours_E - seen["E"]).abs().max()))
    return {"max_abs_diff_X": worst["X"], "max_abs_diff_E": worst["E"], "times": times}


def check_nfe_accounting(den: DiGressDenoiser, n: int = 7, m: int = 6, forward_batch: int = 5) -> dict[str, int]:
    """(c) ``nfe`` after one ``propose`` and one ``predict_x0`` versus graphs seen by the network (forward hook)."""
    old_chunk, den.max_forward_batch = den.max_forward_batch, forward_batch
    calls: list[int] = []
    hook = den.model.model.register_forward_hook(lambda mod, args, out: calls.append(int(args[0].shape[0])))
    try:
        gen = torch.Generator().manual_seed(0)
        state = den.init(n, gen)
        nfe0 = den.nfe
        kids = den.propose(state, min(10, den.num_steps), m, gen)
        after_propose = den.nfe - nfe0
        den.predict_x0(kids, 1)
        total = den.nfe - nfe0
    finally:
        hook.remove()
        den.max_forward_batch = old_chunk
    return {"nfe_propose": after_propose, "expected_propose": n, "nfe_total": total,
            "network_graphs": sum(calls), "max_chunk": max(calls), "chunk_limit": forward_batch}


def official_samples(den: DiGressDenoiser, n: int, batch_size: int = 512) -> list[list[torch.Tensor]]:
    """``n`` molecules from DiGress's own ``sample_batch`` (global RNG; visualisation off) in DiGress's graph format."""
    model, vis = den.model, den.model.visualization_tools
    model.visualization_tools = None                              # sample_batch would otherwise write chain images
    out: list[list[torch.Tensor]] = []
    try:
        with torch.no_grad():
            while len(out) < n:
                b = min(batch_size, n - len(out))
                out += model.sample_batch(batch_id=0, batch_size=b, keep_chain=0, number_chain_steps=1, save_final=0)
    finally:
        model.visualization_tools = vis
    return out


def digress_metrics(den: DiGressDenoiser, graphs: list[list[torch.Tensor]]) -> dict[str, float]:
    """Validity (largest fragment, DiGress's convention) and uniqueness, using DiGress's own metric code."""
    from analysis.rdkit_functions import BasicMolecularMetrics  # noqa: WPS433

    metrics = BasicMolecularMetrics(den.info)
    valid, validity, _, _ = metrics.compute_validity(graphs)
    uniqueness = metrics.compute_uniqueness(valid)[1] if valid else 0.0
    return {"validity": float(validity), "uniqueness": float(uniqueness)}


def compare_with_official_sampler(den: DiGressDenoiser, n: int = 2000, seed: int = 0, batch_size: int = 512,
                                  tol: float = 0.015) -> dict[str, Any]:
    """(b) Validity and uniqueness of ``n`` unguided samples from the wrapper and from DiGress's own sampler.

    Both sample sets go through the same DiGress metric functions. Passes if both gaps are within ``tol`` (the plan's
    1.5 points). Also returns this project's stricter validity (connected only), pass counts and timings.
    """
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    off = official_samples(den, n, batch_size)
    t_off = time.time() - t0
    nfe0, t0, ours_graphs, ours_smiles = den.nfe, time.time(), [], []
    while len(ours_graphs) < n:
        b = min(batch_size, n - len(ours_graphs))
        state = svdd_sample(den, None, b, 1, gen)
        ours_graphs += den.to_graphs(state)
        ours_smiles += den.decode(state)
    t_ours = time.time() - t0
    m_off, m_ours = digress_metrics(den, off), digress_metrics(den, ours_graphs)
    own = compute_metrics(ours_smiles, get_reward("sa"))
    gap = {k: abs(m_off[k] - m_ours[k]) for k in m_off}
    return {"n": n, "official": m_off, "wrapper": m_ours, "gap": gap, "tol": tol,
            "passes": all(g <= tol for g in gap.values()),
            "wrapper_validity_connected": own["validity"], "nfe_per_molecule": (den.nfe - nfe0) / n,
            "expected_nfe_per_molecule": den.num_steps, "seconds_official": t_off, "seconds_wrapper": t_ours}
