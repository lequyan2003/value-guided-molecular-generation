.PHONY: slot-a fallback headline help install test smoke-test splits sweep reference results compile-proposal compile-report clean

help:
	@echo "Targets: install, test, smoke-test, splits, sweep, reference, results, compile-proposal, compile-report, clean"

install:
	pip install -r requirements.txt

test:
	python -m pytest vgmg/tests -v

smoke-test:
	python scripts/smoke_test.py

splits:
	python -m vgmg.data.download && python scripts/prepare_splits.py

sweep:
	python scripts/run_ablations.py $(SWEEP)

compile-proposal:
	cd docs/proposal && latexmk -pdf -interaction=nonstopmode Abel_proposal.tex

compile-report:
	cd docs/report && latexmk -pdf -interaction=nonstopmode Abel_report.tex

clean:
	rm -rf results/logs/* results/smoke results/value_fns
	find docs -name '*.aux' -o -name '*.log' -o -name '*.bbl' -o -name '*.blg' -o -name '*.out' -o -name '*.fls' -o -name '*.fdb_latexmk' | xargs rm -f

SWEEP ?= configs/sweeps/a1_m_sweep.yaml

reference:
	python scripts/qm9_reference.py

results:
	python scripts/generate_results_table.py

fallback:
	python scripts/run_ablations.py configs/sweeps/fallback.yaml

# Headline table, in priority order (see docs/PLAN.md): needs the DiGress wrapper and GPU hours.
headline:
	python scripts/run_ablations.py configs/sweeps/b1_unguided.yaml
	python scripts/run_ablations.py configs/sweeps/a1_m_sweep.yaml
	python scripts/run_ablations.py configs/sweeps/b2_best_of_n.yaml
	python scripts/run_ablations.py configs/sweeps/extra_seeds_svdd.yaml
	python scripts/run_ablations.py configs/sweeps/extra_seeds_b2.yaml
	python scripts/generate_results_table.py

# Slot A talk results in priority order (docs/PLAN.md Sec. 4a): B1, B2 (N = M = 10), SVDD-MLP (M = 10, SA), then PM + random control.
slot-a:
	python scripts/run_ablations.py configs/sweeps/slot_a_b1.yaml
	python scripts/run_ablations.py configs/sweeps/slot_a_b2.yaml
	python scripts/run_ablations.py configs/sweeps/slot_a_svdd.yaml
	python scripts/run_ablations.py configs/sweeps/slot_a_svdd_softmax.yaml
	python scripts/run_ablations.py configs/sweeps/slot_a_controls.yaml
	python scripts/generate_results_table.py
