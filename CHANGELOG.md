# Changelog

## Unreleased

- Add a budget or penalty on case churn or expected negative flips, and match comparisons on
  both queue movement and expected flips.
- Price movement by queue (`queue_costs`), so budgets can count cases, hours, or money.
- Return per-queue dual prices from the flow solver.
- Breaking: `solve_assignment` takes incumbent labels instead of incumbent loads.
- Add a regime simulation mapping when holding queue totals is costly and how learned prices
  compare with exact per-batch assignment.

## 0.1.0 - 2026-08-17

- Define queue shift as incumbent-relative workload movement across predicted queues.
- Add an exact minimum-cost-flow assignment solver and matched-budget guarantee.
- Add oracle and estimated-model validation against NFR probability interpolation.
- Add a reproducible paper, generated exhibits, adversarial checks, and package CI.
