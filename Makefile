PYTHON := uv run python
SOURCE_DATE_EPOCH := 1786960800
export SOURCE_DATE_EPOCH

.PHONY: sync format lint test build theory oracle estimated analyze reproduce check-generated paper research-check check clean

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
	uv run pytest -q

build:
	uv build

theory:
	$(PYTHON) -m experiments.check_theory

oracle:
	$(PYTHON) -m experiments.run_oracle --outdir results/oracle

estimated:
	$(PYTHON) -m experiments.run_estimated --outdir results/estimated

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

reproduce: oracle estimated analyze

check-generated:
	@tmp=$$(mktemp -d) && \
	MPLCONFIGDIR=/tmp/queue-shift-matplotlib $(PYTHON) -m experiments.analyze_oracle \
		results/oracle --out $$tmp/oracle_summary.csv --tex $$tmp/oracle_summary.tex \
		--figure $$tmp/oracle_frontier.pdf --macros $$tmp/oracle_numbers.tex >/dev/null && \
	MPLCONFIGDIR=/tmp/queue-shift-matplotlib $(PYTHON) -m experiments.analyze_estimated \
		results/estimated --out $$tmp/estimated_summary.csv --tex $$tmp/estimated_summary.tex \
		--figure $$tmp/estimated_validation.pdf --macros $$tmp/estimated_numbers.tex >/dev/null && \
	diff -q results/oracle_summary.csv $$tmp/oracle_summary.csv >/dev/null && \
	diff -q results/estimated_summary.csv $$tmp/estimated_summary.csv >/dev/null && \
	diff -q paper/generated/oracle_summary.tex $$tmp/oracle_summary.tex >/dev/null && \
	diff -q paper/generated/estimated_summary.tex $$tmp/estimated_summary.tex >/dev/null && \
	diff -q paper/generated/oracle_numbers.tex $$tmp/oracle_numbers.tex >/dev/null && \
	diff -q paper/generated/estimated_numbers.tex $$tmp/estimated_numbers.tex >/dev/null && \
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
