PYTHON := uv run python
SOURCE_DATE_EPOCH := 1786960800
export SOURCE_DATE_EPOCH

.PHONY: sync format lint test build theory oracle estimated regimes analyze reproduce check-generated paper research-check check clean

sync:
	uv sync --all-groups --all-extras

format:
	uv run ruff check --fix src experiments tests
	uv run ruff format src experiments tests

lint:
	uv run ruff check src experiments tests
	uv run ruff format --check src experiments tests
	uv run pyright
	uvx --from pydoclint==0.9.1 pydoclint src/

test:
	$(PYTHON) -m pytest -q

build:
	uv build

theory:
	$(PYTHON) -m experiments.check_theory

oracle:
	$(PYTHON) -m experiments.run_oracle --outdir results/oracle

estimated:
	$(PYTHON) -m experiments.run_estimated --outdir results/estimated

regimes:
	$(PYTHON) -m experiments.run_regimes --outdir results/regimes

analyze:
	MPLCONFIGDIR=/tmp/queue-shift-matplotlib $(PYTHON) -m experiments.analyze_oracle \
		results/oracle \
		--out results/oracle_summary.csv \
		--tex paper/generated/oracle_summary.tex \
		--figure paper/figures/oracle_frontier.pdf \
		--macros paper/generated/oracle_numbers.tex
	MPLCONFIGDIR=/tmp/queue-shift-matplotlib $(PYTHON) -m experiments.analyze_estimated \
		results/estimated \
		--out results/estimated_summary.csv \
		--tex paper/generated/estimated_summary.tex \
		--figure paper/figures/estimated_validation.pdf \
		--macros paper/generated/estimated_numbers.tex
	MPLCONFIGDIR=/tmp/queue-shift-matplotlib $(PYTHON) -m experiments.analyze_regimes \
		results/regimes \
		--out results/regimes_summary.csv \
		--lock-out results/regimes_lock_cost.csv \
		--lock-tex paper/generated/regimes_lock_cost.tex \
		--tex paper/generated/regimes_offsets.tex \
		--frontier paper/figures/regimes_frontier.pdf \
		--mix-gap paper/figures/regimes_mix_gap.pdf \
		--macros paper/generated/regimes_numbers.tex

reproduce: oracle estimated regimes analyze

check-generated:
	@tmp=$$(mktemp -d) && \
	MPLCONFIGDIR=/tmp/queue-shift-matplotlib $(PYTHON) -m experiments.analyze_oracle \
		results/oracle --out $$tmp/oracle_summary.csv --tex $$tmp/oracle_summary.tex \
		--figure $$tmp/oracle_frontier.pdf --macros $$tmp/oracle_numbers.tex >/dev/null && \
	MPLCONFIGDIR=/tmp/queue-shift-matplotlib $(PYTHON) -m experiments.analyze_estimated \
		results/estimated --out $$tmp/estimated_summary.csv --tex $$tmp/estimated_summary.tex \
		--figure $$tmp/estimated_validation.pdf --macros $$tmp/estimated_numbers.tex >/dev/null && \
	MPLCONFIGDIR=/tmp/queue-shift-matplotlib $(PYTHON) -m experiments.analyze_regimes \
		results/regimes --out $$tmp/regimes_summary.csv --lock-out $$tmp/regimes_lock_cost.csv \
		--lock-tex $$tmp/regimes_lock_cost.tex \
		--tex $$tmp/regimes_offsets.tex --frontier $$tmp/regimes_frontier.pdf \
		--mix-gap $$tmp/regimes_mix_gap.pdf --macros $$tmp/regimes_numbers.tex >/dev/null && \
	diff -q results/oracle_summary.csv $$tmp/oracle_summary.csv && \
	diff -q results/estimated_summary.csv $$tmp/estimated_summary.csv && \
	diff -q paper/generated/oracle_summary.tex $$tmp/oracle_summary.tex && \
	diff -q paper/generated/estimated_summary.tex $$tmp/estimated_summary.tex && \
	diff -q paper/generated/oracle_numbers.tex $$tmp/oracle_numbers.tex && \
	diff -q paper/generated/estimated_numbers.tex $$tmp/estimated_numbers.tex && \
	diff -q results/regimes_summary.csv $$tmp/regimes_summary.csv && \
	diff -q results/regimes_lock_cost.csv $$tmp/regimes_lock_cost.csv && \
	diff -q paper/generated/regimes_offsets.tex $$tmp/regimes_offsets.tex && \
	diff -q paper/generated/regimes_lock_cost.tex $$tmp/regimes_lock_cost.tex && \
	diff -q paper/generated/regimes_numbers.tex $$tmp/regimes_numbers.tex && \
	rm -rf $$tmp && echo "generated results are synchronized"

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null
	cd paper && bibtex main >/dev/null
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null
	@! grep -qE "Undefined control sequence|Citation .* undefined|Reference .* undefined" paper/main.log
	@echo "built paper/main.pdf"

research-check: theory check-generated paper

check: lint test build research-check

clean:
	rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out
