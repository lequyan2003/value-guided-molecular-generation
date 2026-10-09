.PHONY: help install test smoke-test splits sweep reference results compile-proposal compile-report clean

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
	cd docs/proposal && latexmk -pdf -interaction=nonstopmode vgmg_proposal.tex

compile-report:
	cd docs/report && latexmk -pdf -interaction=nonstopmode vgmg_report.tex

clean:
	rm -rf results/logs/* results/smoke results/value_fns
	find docs -name '*.aux' -o -name '*.log' -o -name '*.bbl' -o -name '*.blg' -o -name '*.out' -o -name '*.fls' -o -name '*.fdb_latexmk' | xargs rm -f

SWEEP ?= configs/sweeps/a1_m_sweep.yaml

reference:
	python scripts/qm9_reference.py

results:
	python scripts/generate_results_table.py
