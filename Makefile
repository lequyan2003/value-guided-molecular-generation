.PHONY: slot-a fallback headline help install test smoke-test splits sweep reference results reproduce export-records compile-proposal compile-report clean

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

# No GPU: regenerate the results table, the pre-registered verdict and the figures from the shipped run records.
reproduce:
	python scripts/generate_results_table.py release_results/logs
	python scripts/make_figures.py release_results/logs release_results/figures

# Copy the small final files of every finished run from results/logs to the tracked release_results/logs.
export-records:
	python scripts/export_run_records.py

fallback:
	python scripts/run_ablations.py configs/sweeps/fallback.yaml

# Every experiment of the proposal on a GPU box with the frozen DiGress checkpoint (see docs/PLAN.md Sec. 4c). Finished runs are skipped.
# B1 seeds 0-4, B2 pool N in {5,10,20}, SVDD-MLP / PM / random control, A1 (M sweep), A2 (GNN), seeds 3-4 at M = 10.
headline:
	python scripts/run_ablations.py configs/sweeps/b1_unguided.yaml
	python scripts/run_ablations.py configs/sweeps/b2_best_of_n.yaml
	python scripts/run_ablations.py configs/sweeps/a1_m_sweep.yaml
	python scripts/run_ablations.py configs/sweeps/a2_estimator.yaml
	python scripts/run_ablations.py configs/sweeps/a2_gnn.yaml
	python scripts/run_ablations.py configs/sweeps/extra_seeds_svdd.yaml
	python scripts/run_ablations.py configs/sweeps/extra_seeds_b2.yaml
	python scripts/generate_results_table.py

# Exploratory (not in the proposal): H4 refit of the value function on its own guided rollouts; softmax selection arm.
exploratory:
	python scripts/run_ablations.py configs/sweeps/h4_refit.yaml
	python scripts/run_ablations.py configs/sweeps/slot_a_svdd_softmax.yaml

# Slot A talk results in priority order (docs/PLAN.md Sec. 4a): B1, B2 (N = M = 10), SVDD-MLP (M = 10, SA), then PM + random control.
slot-a:
	python scripts/run_ablations.py configs/sweeps/slot_a_b1.yaml
	python scripts/run_ablations.py configs/sweeps/slot_a_b2.yaml
	python scripts/run_ablations.py configs/sweeps/slot_a_svdd.yaml
	python scripts/run_ablations.py configs/sweeps/slot_a_svdd_softmax.yaml
	python scripts/run_ablations.py configs/sweeps/slot_a_controls.yaml
	python scripts/generate_results_table.py
