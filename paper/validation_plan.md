# Estimated-model validation design

This document records the validation implemented in `experiments/run_estimated.py`. It is a
reproducibility aid, not a preregistration. The simulations are supporting evidence for a
mathematical result, not evidence about the effect size in any organization.

## Question and comparison

The data distribution is stable. An incumbent classifier uses one historical signal block. A
candidate uses the incumbent block plus a new signal block. The candidate is released only if it
is more accurate on an independent labeled release sample.

On later, unlabeled planning batches, the comparison policy interpolates the incumbent and
candidate probabilities. This is the strong negative-flip-rate baseline. For every interpolation
weight, Queue Shift receives exactly the queue-movement budget used by the baseline. Both policies
are scored only after their assignments are fixed.

The main reported quantity is the mean accuracy difference, in percentage points, between Queue
Shift and interpolation at the same queue movement. Planning batches are averaged within each
independently trained model pair before uncertainty is computed. Results at interpolation weights
0 and 0.5 receive the most attention. Weight 1 is a required equality check because it is the raw
candidate endpoint.

## Models and scenarios

- Five queues have probabilities `(0.40, 0.25, 0.15, 0.12, 0.08)`.
- Historical, release, and planning samples come from one fixed Gaussian data-generating process.
- Both fitted models are multinomial logistic regressions.
- Each scenario uses 50 trained model pairs, a release sample of 2,000 cases, four planning batches
  of 300 cases, and interpolation weights `(0, 0.25, 0.5, 0.75, 1)`.
- `well_estimated` uses 5,000 historical cases and innovation signal 0.8.
- `small_history` uses 500 historical cases and innovation signal 0.8.
- `weak_innovation` uses 2,000 historical cases and innovation signal 0.3.
- `overconfident` uses 2,000 historical cases, innovation signal 0.8, and candidate temperature
  0.5.
- `underconfident` uses 2,000 historical cases, innovation signal 0.8, and candidate temperature
  2.

Temperature scaling changes the probability values used by the optimizer but not the candidate's
hard predictions or release decision.

## Checks

The code verifies four statements without sampling tolerance beyond numerical precision:

1. Queue Shift never exceeds the baseline's queue movement.
2. Queue Shift never has lower value under the supplied candidate probabilities.
3. The raw-candidate endpoint has zero supplied-score gain.
4. The probability-error lower bound never exceeds the true conditional expected gain.

The analysis reports repetition-level means and 95% normal intervals for conditional expected and
realized accuracy. Acceptance rates are reported because rejected candidate models do not enter
planning batches. A scenario is never removed because its result is weak or negative.
