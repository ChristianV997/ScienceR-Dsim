# ITCT Cessation Protocol v3 — Figure Captions

**Status: pending real-data figure generation.**

Figures for this protocol are produced by running
`analysis/itct/itct_cessation_protocol_v3_full_stack.py` against a real BIDS dataset
(e.g. ds005620) and plotting the per-window `beta1`, `spectral_dimension`, and
`loschmidt_echo` series from `itct_cessation_result.json`.

Captions here are intentionally not pre-written: they must describe real figures produced
from a real run with `provenance=real_bids`, not synthetic proxies. Generate the figures
first, then write captions that state the dataset, subject, task, window length, and the
`not_validated_here` caveats carried in the result JSON.
