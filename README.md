# Queue Shift

Classifier updates should respect the cost they actually create. When predictions route cases to
specialized queues, that cost is the change in queue totals, not the number of individual
predictions that change.

Queue Shift finds the highest-value batch assignment under two limits: on work moved between
queues, counted in cases, hours, or money, and on expected negative flips. With only the movement
limit, or with a penalty on flips, the problem is a minimum-cost flow with an integral global
optimum and per-queue prices that reproduce it case by case; hard flip budgets and per-queue
prices are solved exactly as small integer programs. At the movement and expected flips of any
comparison rule, including a negative-flip interpolation, Queue Shift has at least the rule's total
probability score. If those probabilities are the true conditional probabilities, the result is
dominance in conditional expected accuracy.

The paper proves the result and uses simulation to map where the constraint matters. Holding queue
totals costs little when an update mostly reorders cases and a lot when it moves volume between
queues. Learned queue prices nearly match exact assignment in accuracy but miss the movement budget
in most batches.

Read the current manuscript: [Queue Shift: Updating Classifiers Under a Staffing
Budget](paper/main.pdf).

## Install

```bash
uv sync --all-groups --all-extras
```

## Use the solver

```python
import numpy as np

from queue_shift import case_flip_weights, solve_assignment

candidate_probability = np.array(
    [
        [0.80, 0.15, 0.05],
        [0.25, 0.65, 0.10],
        [0.20, 0.30, 0.50],
    ]
)
incumbent_labels = np.array([0, 2, 1])

shift_only = solve_assignment(
    costs=1 - candidate_probability,
    incumbent_labels=incumbent_labels,
    move_budget=0,
)
print(shift_only.labels, shift_only.churn)  # [0 1 2] 2: a swap keeps totals

weights = case_flip_weights(
    candidate_probability, incumbent_labels, "expected_negative"
)
flip_limited = solve_assignment(
    costs=1 - candidate_probability,
    incumbent_labels=incumbent_labels,
    move_budget=0,
    flip_weights=weights,
    flip_budget=0.2,
)
# [0 2 1] 0.0: the swap would risk 0.1 + 0.3 = 0.4 expected negative flips
print(flip_limited.labels, flip_limited.flip_load)
```

The movement budget limits work added to queues beyond their incumbent loads. By default work is
counted in cases, and the budget equals half the L1 distance between the proposed and incumbent
queue-load vectors. A budget of zero preserves every incumbent queue total while allowing different
cases to fill those slots. Pass `queue_costs` to price a case in each queue, for example in average
handle hours or cost per case; the budget is then in hours or money:

```python
hours_per_case = np.array([0.5, 1.0, 4.0])
by_hours = solve_assignment(
    costs=1 - candidate_probability,
    incumbent_labels=incumbent_labels,
    move_budget=2.0,
    queue_costs=hours_per_case,
)
print(by_hours.labels, by_hours.workload_shift)  # [0 1 2] 0.0
```

Flip weights price a case leaving its incumbent queue. `"expected_negative"` weights a move by
the candidate's probability that the incumbent was right, so the weighted total is the expected
number of negative flips; `"churn"` counts every changed label. `flip_budget` caps the total
exactly (solved as a mixed-integer program). `flip_penalty` instead charges for it, which keeps
the problem a minimum-cost flow, scales to large batches, and returns per-queue prices
(`queue_prices`) that reproduce the assignment case by case.

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
