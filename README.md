# Queue Shift

Classifier updates should respect the cost they actually create. When predictions route cases to
specialized queues, that cost is the change in queue totals, not the number of individual
predictions that change.

Queue Shift finds the highest-value batch assignment under a limit on queue-load movement. The
problem is a minimum-cost flow, so the solver returns an integral global optimum. At the movement
budget produced by any comparison rule, including a negative-flip method, Queue Shift weakly
improves the accepted model's total probability score. If those probabilities are the true
conditional probabilities, the result is dominance in conditional expected accuracy.

The paper proves the result and reports two stable-distribution experiments. The main validation
estimates both classifiers on finite historical samples, releases the candidate only after an
independent accuracy win, and tests small samples, weak innovation, overconfidence, and
underconfidence.

Read the current manuscript: [Queue Shift: Updating Classifiers Under a Staffing
Budget](paper/main.pdf).

## Install

```bash
uv sync --all-groups --all-extras
```

## Use the solver

```python
import numpy as np

from queue_shift import solve_assignment

candidate_probability = np.array(
    [
        [0.80, 0.15, 0.05],
        [0.25, 0.65, 0.10],
        [0.20, 0.30, 0.50],
    ]
)
incumbent_loads = np.array([1, 1, 1])

result = solve_assignment(
    costs=1 - candidate_probability,
    incumbent_loads=incumbent_loads,
    move_budget=0,
)
print(result.labels)
```

The movement budget is half the L1 distance between the proposed and incumbent queue-load
vectors. A budget of zero preserves every incumbent queue total while allowing different cases to
fill those slots.

## Reproduce the paper

```bash
make reproduce
make paper
```

`make reproduce` reruns the oracle and estimated-model experiments and regenerates every table,
macro, and figure used by the manuscript. `make check` runs formatting checks, linting, tests,
formal adversarial checks, generated-output checks, and the paper build.

## Repository layout

```text
src/queue_shift/     Solver, evaluation, and simulation library
experiments/         Reproducible experiment and analysis entry points
tests/               Unit, brute-force, and adversarial tests
results/             Seeded raw results and computed summaries
paper/               Manuscript, generated tables, and figures
```

The current evidence is simulation based. The optimization guarantee is exact for the supplied
scores, while realized accuracy depends on probability quality. The paper states that distinction
and gives a probability-error bound.
