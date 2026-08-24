# Pre-registered success criteria for the reusability meta-objective

Fixed 2026-08-11. Revised 2026-08-14 — still before any meta-training code is
written. The revision tightens definitions that were ambiguous enough to move
once results existed; it does not relax any threshold.

The oracle experiments established that something is lost on revisit (35/35
DL3DV scenes at 4.47 dB, 90/90 nuScenes scenes at 5.79 dB) and that adapted
states are distinguishable from one another (a correction built from an
unrelated scene costs 1.04 dB where a random one costs 0.06). They also
established that none of it is currently recoverable: a rank-8 correction toward
the remembered state returns 1.6% of the gap, and the difference between two
adapted states has stable rank 43.9.

The project's claim is that this gap exists *because* the inner loop is trained
greedily, and that training for reusability closes it. That claim is close to
unfalsifiable after the fact — any failure can be explained as "not trained for
it yet" — so the numbers that would count as success are fixed here, in
advance, with the measurement procedure attached to each.

## Criteria

| # | Metric | Current | Target | Measured by |
|---|--------|---------|--------|-------------|
| 1 | Recovery rate — see the definition below, which is not the one the 1.6% was computed under | 1.6% (old denominator) | **≥ 20%** | `run_correction.py`, lambda sweep, A-region delta over the bank-off interference |
| 2 | Stable rank of `W_A - W_AB` | 43.9 | **≤ 20** | `run_correction.py` E4 block |
| 3 | Foreign penalty (wrong retrieval must hurt) | −1.037 dB | **maintained or larger** | `run_controls.py`, foreign vs random at matched norm |
| 4 | Share of *memory gain* originating in the stored layers | **not measured** | **≥ 74%** | `run_layer_groups.py` with the memory direction |

### 1. Recovery rate — denominator fixed here

The 1.6% was measured against the interference of the **greedily trained**
model. Meta-training changes the inner loop, so that interference will itself
change, and a ratio whose denominator moves with the treatment measures nothing.

    denominator = interference of the *same trained model*, bank disabled
    numerator   = quality recovered when the bank is enabled

This separates two outcomes that must not be conflated: training may reduce
interference on its own, which would mean the bank is unnecessary, and that is a
failure mode of this project rather than a success. Reporting requires **both**
numbers — the bank-off interference and the recovered fraction — never the ratio
alone.

Against the current 4.5–5.8 dB, returning 20% is about 1 dB, large enough to
survive a parameter-matched baseline. Below that the method is a rounding error
on the quantity it claims to fix.

### 2. Stable rank

The bank stores rank-r factors. At rank 8 the current difference retains 29–45%
of its energy, and E5 showed that *raising* the rank makes things worse, not
better (r128 and full-rank corrections degrade at lambda = 0.05 where rank 8
improves). So the rank cannot be bought with storage; meta-training has to
produce differences that are genuinely low-rank. Target 20 is roughly half the
current value and about twice rank 8.

### 3. Foreign penalty

This guards the failure mode the objective is most likely to fall into. If
meta-training collapses skills toward a regime-independent constant — the
degenerate optimum of a pure reuse objective — a correction from an unrelated
scene stops being harmful, because every skill has become the same skill. A
shrinking foreign penalty is the observable signature of that collapse, and it
is watched from the first training run, not diagnosed afterwards.

### 4. Layer concentration

The 18 MB design stores three layers. That choice rests on where a *wrong*
correction does damage, because the foreign penalty is 23x larger than the
memory gain and is the only signal currently strong enough to localise. Whether
the memory *gain* lives in the same layers is an assumption: at +0.045 dB it
does not decompose (single-layer contributions sum to −0.032, the wrong sign).

The layers were also localised on the greedily trained model — the very thing
being changed. If the gain does not concentrate there once it is large enough to
localise, the storage design is holding the wrong layers. **Run this check
before any other analysis of a training result**, since a rise in recovery rate
is exactly the kind of good news that buries it.

## Design decisions fixed in advance

Locked 2026-08-14, before implementation. Each is here because it is the sort of
choice that drifts toward whatever the results need.

**1. Regime descriptor is pose-only.** Per-frame translation and rotation
statistics over a trailing window — speed, its spread, yaw rate, its spread,
stationary fraction. **No vehicle telemetry.** CAN bus was tested against it on
the 756 pairwise penalties and adds nothing: pose-only reaches rho = −0.336
against CAN bus's −0.316, and once pose is controlled for, CAN bus contributes
−0.038 (p = 0.29) while pose still contributes −0.122 (p = 0.0008) in the other
direction. Dropping it removes a dataset-specific dependency and lets the method
apply wherever a pose stream exists.

Spatial proximity is **not** part of the descriptor: at rho = +0.021 (p = 0.56)
it predicts compatibility not at all, which is the finding the whole design
rests on.

**2. The bank is built during training and frozen at deployment.** Write
requires knowing whether a stored skill helped, which needs the counterfactual
of not having stored it — unavailable at inference. Freezing costs online
adaptation and buys three things: runs are reproducible, an evaluation stream
cannot contaminate the bank, and the comparison against a static prototype
baseline is fair. That baseline is this project's closest competitor, and it
cannot be beaten by a method that alone gets to keep learning at test time.

**3. Read is top-1 with a straight-through estimator.** A softmax mixture would
leave "the retrieved skill" undefined, and the contrastive objective is stated
over a retrieved skill against alternatives. Top-1 keeps one slot equal to one
skill, so the number of skills actually learned remains countable.

**4. The bank uses EMA consolidation with dead-slot reinitialisation.**
Nearest-slot EMA alone collapses the way a VQ-VAE codebook does: slots that
happen to start close absorb everything, the rest stay at their initial values.
Top-1 reading makes this worse, so usage tracking and revival of unused slots
ship with it rather than being added when collapse appears.

**5. The stored layer set is configuration, not a constant.** L1, L2 and L7 came
from a model this work is about to change. Changing the set must not require
retraining anything but the bank.

**6. Trained components.** Regime encoder, bank keys and values, read/write
gates, **and the TTT layer itself** — its inner learning rate and projections.
The last is the mechanism: deltas become reusable because the update learns to
produce them that way, not because the bank compresses them well afterwards.
Without it this is a LoRA bank with a router.

## Compounding losses to keep in view

Storing three layers keeps 74% of the rank-8 effect, and rank 8 itself holds
about 30% of the adaptation energy, so the design as it stands operates on
roughly 22% of what the full state contains. Criterion 2 is what buys that back:
a genuinely low-rank difference makes the rank-8 truncation cheap rather than
lossy.

## Failure routes, decided in advance

- **Criterion 1 misses but 2 and 3 hold.** Retrieval still discriminates and the
  geometry improved, but the correction does not pay. Extend the objective's
  horizon (more re-occurrences per episode) before abandoning; do not relax the
  20% threshold.
- **Bank-off interference falls but recovery stays flat.** Training fixed the
  problem on its own and the bank is redundant. This is a negative result for
  the project and must be reported as one, not as an improvement.
- **Criterion 3 degrades.** Objective collapse. Stop and fix the objective —
  this is not a tuning problem, and continuing produces a method that cannot be
  distinguished from a lower learning rate.
- **Criterion 4 misses.** Redo layer selection with the memory direction and
  re-price the bank. The storage argument survives; only which layers changes.
