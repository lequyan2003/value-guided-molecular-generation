"""GPU-hour estimate in the proposal's form: (steps x s/step)/3600 x runs, with the corrected pass counts.

usage: python scripts/estimate_gpu_hours.py [seconds_per_batch_step=0.1] [overhead=1.5] [T=500] [N=2000] [batch=512]
Re-run with the value MEASURED at gate G1 and put the output in the report's compute section.
Pass counts per molecule (forward passes of the denoiser): unguided T; best-of-m m*T; SVDD-value T;
SVDD-PM (m+1)*T; random-select T. Value-function training: rollouts of train+val chains + fitting (fit_h each).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_ablations import expand_sweep  # noqa: E402


def main() -> None:
    """Print hours per sweep and in total."""
    a = [float(x) for x in sys.argv[1:]] + [0.1, 1.5, 500, 2000, 512][len(sys.argv) - 1:]
    sec, over, T, N, B = a[0], a[1], int(a[2]), int(a[3]), int(a[4])
    fit_h = 0.5 / 1.0  # proposal: 0.5 h per value-function fit (replace with the measured value)
    hours = lambda passes_per_mol, n=N: n * passes_per_mol / B * sec * over / 3600  # noqa: E731
    total, vf_done = 0.0, set()
    for sweep in sorted(Path("configs/sweeps").glob("*.yaml")):
        if sweep.stem == "fallback":
            continue
        h = 0.0
        for cfg in expand_sweep(str(sweep)):
            m, meth, n = cfg["svdd"]["num_candidates_m"], cfg["method"], cfg["eval"]["num_molecules"]
            if meth == "best_of_n":
                h += hours(max(cfg["best_of_n"]["m_values"]) * T, n)      # one pool per seed
            elif meth == "unguided" or meth == "random_select":
                h += hours(T, n)
            elif meth == "svdd_pm":
                h += hours((m + 1) * T, n)
            elif meth == "svdd_value":
                h += hours(T, n)
                key = (cfg["svdd"]["reward_type"], cfg["svdd"]["value_fn_arch"], cfg["seed"])
                if m > 1 and key not in vf_done:
                    vf_done.add(key)
                    vc = cfg["value_fn"]
                    h += hours(T, vc["train_chains"] + vc["val_chains"]) + fit_h
        print(f"{sweep.stem:<22}{h:7.1f} h")
        total += h
    print(f"{'TOTAL':<22}{total:7.1f} h   (+25% rerun margin: {total * 1.25:.1f} h)  "
          f"[s/step={sec}, overhead={over}, T={T}, N={N}, batch={B}]")


if __name__ == "__main__":
    main()
