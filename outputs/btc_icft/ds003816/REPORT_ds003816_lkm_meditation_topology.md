# ds003816 (Loving-Kindness Meditation EEG) — Whole-Montage Real-Signal Topology Port

Full-montage, real-signal (not hash-fabricated) topology extraction ported from the
corrected ds005620/ds003969 pipeline, streamed over all 48 subjects, meditation vs
resting. **Verdict up front: the winding-charge topology instrument detects NO
loving-kindness-meditation-vs-resting signature (paired, within-subject, all five
metrics p ≥ 0.25, |dz| ≤ 0.18) — a genuine null. But the classical Level M
signal features (spectral power, entropy, Lempel-Ziv complexity) DO separate the
two states (paired, all p ≤ 0.003, dz ≈ −0.4 to −0.6, all lower in meditation).
This is the cleanest dissociation across the four datasets ported so far: a
contemplative-state contrast that classical spectral/complexity instruments
resolve and the topology instrument does not — direct evidence that this repo's
q_net/q_abs topology is orthogonal to, and less sensitive than, classical
features for this kind of contrast.**

---

## Part 0 — Provenance & Methodology

- **Dataset:** OpenNeuro `ds003816` (Sun, Wong & Gao — "The Effect of Buddhism
  Derived Loving Kindness Meditation on Modulating EEG", CC0). 48 participants;
  a subset are long-term practitioners (subject IDs ending `lt`, recorded across
  up to 10 sessions), the rest short-term (`st`, single session). 128-channel
  actiCHamp + 1 ECG, 1000 Hz, BrainVision format.
- **Confirmed real BIDS task labels** (direct S3 listing of `sub-01lt` sessions,
  not assumed): `LKMSelf`, `LKMOther` (loving-kindness meditation directed at
  self / other), `PreResting`, `PostResting` (resting baselines), `VisualizeSelf`,
  `VisualizeOther` (a visualization control condition).
- **State mapping:** `LKMSelf`/`LKMOther` → `meditation`; `PreResting`/`PostResting`
  → `resting`. **`VisualizeSelf`/`VisualizeOther` are deliberately left unmapped
  and excluded from the 2-class contrast** — a visualization control is neither
  loving-kindness meditation nor a passive resting baseline, and inventing a label
  for it would violate the `no_label_inference` guardrail. (These windows are
  extracted and stored with `state_label=null`; they simply do not enter the
  contrast.)
- **Pipeline:** the consolidated real-signal pattern — `read_window_signal` →
  `extract_level_m_features` → `compute_topology_from_channels` — reusing the
  dataset-agnostic modules unmodified. New code is thin per-dataset shims
  (`level_m/ds003816_windows{,_real}.py`, `level_t/ds003816_real_topology.py`,
  `pipelines/run_ds003816_{m,t}_real.py`, `configs/btc_icft/ds003816_{m,t}_real.yaml`)
  plus a `process_ds003816_subject` entry in the streaming runner — exactly the
  "a new dataset should cost a thin shim, not a from-scratch port" outcome the
  repo-hardening consolidation was built for.
- **Windowing:** 4 s windows, up to 5 per recording, 8 channels post-selection
  (topology-triangulation tractability). **Cohort: all 48 subjects streamed from
  S3, peak disk bounded to one subject at a time, 0 subject-level failures,
  5385 total M+T window rows** (meditation 1805, resting 1790, unmapped Visualize
  1790).

### A real bug this port surfaced (now fixed, affects all datasets)

Several signal-read call sites caught only `ValueError` (out-of-range window),
not `OSError`. A BrainVision `.vhdr` whose companion `.eeg`/`.vmrk` is genuinely
missing from the dataset's own S3 bucket — confirmed for real on `sub-01lt`/
`sub-03lt` session `ses-03`, one task each — raises `FileNotFoundError` while
mne parses the header. Before the fix, one such incomplete recording crashed the
entire per-subject batch and discarded every other readable window from that
subject; the streaming runner then marked the whole subject failed. Now
skip-and-report (a warning + a NaN/zeroed row for just that window), matching the
existing out-of-range handling. This is why all 48 subjects completed cleanly
despite several having a missing companion file.

---

## Part 1 — Primary Result: Topology by State (Paired, Within-Subject)

Every subject is measured in both conditions, so the correct unit of analysis is
the per-subject paired difference, not a window-pooled or between-subjects test
(subject-blocked paired sign-flip permutation, 5000 shuffles, seed 0; effect size
is paired Cohen's dz). 47 of 48 subjects contributed data to both conditions.

| Metric | meditation (mean±SD) | resting (mean±SD) | dz (paired) | p (paired) |
|---|---|---|---|---|
| q_net | 0.00430 ± 0.01232 | 0.00445 ± 0.01267 | +0.083 | 0.727 |
| q_abs | 0.00754 ± 0.01274 | 0.00766 ± 0.01308 | +0.001 | 0.996 |
| f_dress | 0.00324 ± 0.00259 | 0.00320 ± 0.00281 | −0.181 | 0.246 |
| defect_density | 0.00013 ± 0.00023 | 0.00014 ± 0.00023 | +0.001 | 0.996 |
| topology_quality | 0.8637 ± 0.343 | 0.8654 ± 0.341 | −0.118 | 0.443 |

**Every topology metric is null.** No metric approaches significance; the largest
effect (f_dress, dz = −0.18, p = 0.25) is a small non-significant trend. This is
not a partial or borderline result — the winding-charge topology instrument does
not separate loving-kindness meditation from resting.

---

## Part 2 — Secondary Result: Classical Level M Features DO Separate the States

The same subjects, same windows, same paired test — applied to the classical
signal-level features the topology metrics sit alongside (44 subjects had real
per-condition data for these):

| Level M feature | meditation | resting | dz (paired) | p (paired) |
|---|---|---|---|---|
| spectral_power_proxy | 0.000259 | 0.000376 | −0.405 | 0.001 |
| entropy_proxy | 5.061 | 5.084 | −0.597 | <0.001 |
| lzc_proxy (complexity) | 0.005631 | 0.005753 | −0.483 | 0.003 |

**All three classical features are significantly LOWER in meditation than in
resting** (medium paired effect sizes), and all survive the pseudoreplication-safe
paired test — this is not a window-pooling artifact. Lower spectral power, lower
amplitude-histogram entropy, and lower Lempel-Ziv complexity during loving-kindness
meditation relative to rest is directionally consistent with the broad EEG
meditation literature (reduced cortical activation / complexity, increased
low-frequency regularity during focused practice). These are the repo's own
labelled `_proxy` heuristics, not validated clinical biomarkers, and are reported
here as measured values against the literature's direction, nothing more.

---

## Part 3 — Interpretation & Cross-Dataset Pattern

The dissociation is the finding. Within a single dataset, on identical windows and
an identical paired design:

- **classical spectral/complexity features:** medium, significant state effect;
- **winding-charge topology (q_net/q_abs/f_dress/defect_density/quality):** flat null.

So the state difference here lives in the **absolute spectral/complexity magnitude
of the signal, not in the topological structure** the channel-mean/correlation
instrument measures. The topology instrument is genuinely orthogonal to — and, for
this contrast, strictly less sensitive than — the classical features.

Placed against the three prior ports:

- **ds005620 (propofol sedation):** a large discrete pharmacological transition —
  topology *moved* (the instrument's one positive result).
- **ds003969 (meditation vs thinking):** subject-controlled behavioral state switch
  — topology null, and classical features also largely null.
- **ds001787 (expert vs novice, between-subjects):** trait contrast, non-significant
  trends once corrected to subject-level.
- **ds003816 (LKM meditation vs resting):** topology null **but classical features
  significant** — the first port where the two instrument families clearly
  disagree.

The pattern that survives all four: this repo's winding-charge topology responds to
large discrete pharmacological state transitions (ds005620) and not to
subject-controlled contemplative/behavioral state switches (ds003969, ds003816),
regardless of whether classical spectral features do.

---

## Part 4 — Confounds & Limitations

- **Artifact quality is comparable across states** (mean artifact_score ≈ 0.28
  meditation vs 0.27 resting) — the topology null is not masking a signal-quality
  difference between conditions, and the classical-feature effect is not a pure
  artifact-level difference either.
- **Window-count balance:** near-balanced (meditation 1805, resting 1790), and the
  paired design handles residual per-subject imbalance by construction.
- **`_proxy` features are heuristics**, not FFT band power / permutation entropy /
  true Lempel-Ziv — the real-instrument replacements (`real_features.py`) exist and
  could sharpen the Level M result; this report uses the legacy proxies for
  cross-dataset comparability with the earlier ports.
- **8-channel, 4-s windowing** is a tractability constraint; the topology instrument
  operates on channel means and a correlation-threshold triangle count, not on
  band-specific phase or genuine spatial winding — its null here is a statement
  about *this* instrument, not about whether any topological signature exists.
- **`lt`/`st` (long-term / short-term practitioner) structure is not analyzed here**
  — this is a state (meditation vs rest) contrast, not a trait/expertise contrast;
  the practitioner-experience question is left as future work.

---

## Verdict

One sentence: **loving-kindness meditation vs resting produces a clear, paired,
pseudoreplication-safe difference in classical EEG spectral/complexity features
(all lower in meditation, p ≤ 0.003) but no detectable difference in winding-charge
topology (all p ≥ 0.25) — matching ds003969's "topology is null for a
subject-controlled contemplative-state switch" pattern, and adding the sharpest
demonstration yet that this repo's topology instrument is orthogonal to, and less
sensitive than, classical spectral features for a contemplative-state contrast.**
