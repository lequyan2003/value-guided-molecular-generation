"""Gate for the DiGress wrapper: exactness tests (a)-(c) on a trained checkpoint (docs/PLAN.md, Stage 2; risk R4/R19).

    python scripts/verify_digress_wrapper.py --checkpoint path/to/last.ckpt [--n 2000] [--out results/digress_checks.json]

Run inside the DiGress environment on a GPU. No experiment result may be produced from a wrapper that fails this script
(exit code 1). The report also holds DiGress-metric validity and uniqueness of the unguided samples (decision D2:
validity >= 90 %, target >= 97 %), which is why test (b) is run on the frozen model and not only on early checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.models.digress import DiGressDenoiser  # noqa: E402
from vgmg.models.digress_checks import (  # noqa: E402
    check_nfe_accounting, check_posterior_exactness, compare_with_official_sampler)

ATOL_POSTERIOR = 1e-6  # plan, test (a)


def sha256(path: Path) -> str:
    """SHA-256 of a file (the checkpoint identity written into every report and run record)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    """Run the three checks, print a summary, write the JSON report; return the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--n", type=int, default=2000, help="molecules per sampler for test (b)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/digress_checks.json"))
    args = ap.parse_args()

    den = DiGressDenoiser(args.checkpoint, args.device)
    a = check_posterior_exactness(den, seed=args.seed)
    c = check_nfe_accounting(den)
    b = compare_with_official_sampler(den, args.n, args.seed, args.batch_size)
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "device": str(den.device),
        "diffusion_steps": den.num_steps,
        "a_posterior": {**a, "atol": ATOL_POSTERIOR,
                        "passes": max(a["max_abs_diff_X"], a["max_abs_diff_E"]) <= ATOL_POSTERIOR},
        "b_official_sampler": b,
        "c_nfe": {**c, "passes": c["nfe_propose"] == c["expected_propose"] and c["nfe_total"] == c["network_graphs"]
                  and c["max_chunk"] <= c["chunk_limit"]},
    }
    report["all_pass"] = all(report[k]["passes"] for k in ("a_posterior", "b_official_sampler", "c_nfe"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    o, w = b["official"], b["wrapper"]
    print(f"(a) posterior max |diff|  X {a['max_abs_diff_X']:.2e}  E {a['max_abs_diff_E']:.2e}  -> {report['a_posterior']['passes']}")
    print(f"(b) official  validity {o['validity']:.3f} uniqueness {o['uniqueness']:.3f}")
    print(f"    wrapper   validity {w['validity']:.3f} uniqueness {w['uniqueness']:.3f}  "
          f"(connected-only validity {b['wrapper_validity_connected']:.3f}) -> {b['passes']}")
    print(f"    s/{b['n']} molecules: official {b['seconds_official']:.0f}, wrapper {b['seconds_wrapper']:.0f}; "
          f"nfe/molecule {b['nfe_per_molecule']:.0f} (T = {b['expected_nfe_per_molecule']})")
    print(f"(c) nfe {c['nfe_total']} vs network graphs {c['network_graphs']} -> {report['c_nfe']['passes']}")
    print("ALL PASS" if report["all_pass"] else "FAILED: no result may be produced from this wrapper", f"-> {args.out}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
