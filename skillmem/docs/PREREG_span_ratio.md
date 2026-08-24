# Pre-registration — segment span ratio and interference

Registered 2026-08-12, before the confirmatory scenes were selected or converted.

## Where this came from

Thirty nuScenes scenes were measured to check the model could be used on driving
data at all. It could, and interference came out at 4.5 dB, matching DL3DV. Two
scenes then stood out at 13.4 and 11.9 dB with almost no moving objects in
frame, so traffic was not the cause; the CAN bus showed both had spent most of
the A segment stationary and then pulled away.

That reading is **exploratory**. Twenty-five driving statistics were screened
against one outcome, nothing survived correction at n = 30, and the six scenes
quoted were picked by eye after the fact. It is a hypothesis, not a result, and
it is registered here so the test is not run on the data that produced it.

## The mechanism being claimed

While the car is stopped the camera barely moves, so a segment spent stationary
presents the model with eight nearly identical views. The fast weight
specialises to a very narrow distribution of viewpoints. When the car then pulls
away and the viewpoints spread out, that specialisation has nothing left to
apply to and is overwritten wholesale.

The claim is therefore about **how much wider B's viewpoints are than A's**, not
about stopping. Stopping is the extreme case where A's span approaches zero.

## Primary hypothesis

**H1.** Interference increases with `span_ratio = span_B / span_A`, where
`span_X` is the maximum pairwise distance between camera positions within
segment X, in the shared normalised frame. Tested as a partial Spearman
correlation controlling for the A-only PSNR.

Predicted sign: **positive**. Threshold: q < 0.05 after correction over the
registered tests only.

## Secondary hypothesis, and why it exists

`a_stopped_frac` cannot be tested cleanly because it sits upstream of the
control variable: a stationary A segment is easy to reconstruct, which raises
the A-only PSNR, which independently predicts interference through a pure
ceiling effect (rho ~ 0.64). Controlling for PSNR removes part of the cause;
not controlling for it lets the ceiling effect in. No sample size fixes this.

`span_ratio` breaks the tie because it runs in both directions.

**H2.** Among scenes where `span_ratio < 1` — A driving, B slowing or stopping —
interference is lower than among scenes where `span_ratio > 1`. In this subset
the A segment is *harder* to reconstruct, so the ceiling effect pushes
interference the opposite way from H1. If H1 holds and H2 also holds, the
ceiling effect cannot be the explanation. If H1 holds and H2 fails, it probably
is.

## Design

- **Scenes.** 90 drawn at random from the 850, stratified across the four
  locations in proportion to their scene counts, **excluding every scene already
  measured**. The exploratory thirty are not reused: they generated the
  hypothesis, and testing on them would be circular.
- Random draw rather than consecutive numbering: nuScenes scene numbers track
  recording sessions, so a contiguous block would concentrate in a few sessions
  and locations.
- **Protocol.** Identical to the exploratory run — CAM_FRONT at 12 Hz, 8 input
  views per segment, `a_span = 0.30`, 4 held-out targets in A's region, no
  masking (dynamic objects were shown to account for 2.5% of interference).
  Targets are drawn from annotated keyframes. This is a carry-over from the
  masking experiment, where scored frames had to have boxes; masking is not
  applied here and the restriction is no longer needed, but it is kept so that
  the confirmatory run scores the same kind of frame as the exploratory one it
  is testing. Corrected on the day of the run, before any of these scenes'
  results were examined.
- **Registered tests: two.** H1 and H2, nothing else. Any other statistic
  computed from these scenes is exploratory and will be labelled as such.

## What each outcome means

| H1 | H2 | Reading |
|----|----|---------|
| holds | holds | Mechanism confirmed. `span_ratio` becomes the regime axis, and scene selection for the 2x2 can be automated over all 850. |
| holds | fails | Probably the ceiling effect wearing a different name. Do not build the regime axis on it. |
| fails | — | The exploratory pattern was six scenes of noise. Fall back to choosing scenes by hand. |

## Why this variable and not the CAN bus one

`a_stopped_frac` is a nuScenes-only quantity. `span_ratio` is the same family as
`spread_a` and `parallax_ratio`, which were already measured on DL3DV — where
the within-scene partial correlation with interference was +0.377 for
`parallax_ratio` and −0.302 for `spread_a`. Using it keeps the two datasets in
one language, so a positive result here can be read directly against the DL3DV
numbers rather than standing alone.
