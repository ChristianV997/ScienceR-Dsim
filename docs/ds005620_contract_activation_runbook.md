# DS005620 Label Contract Activation Audit Runbook (P16)

---

## Purpose

P16 produces a read-only audit of local DS005620 metadata files to determine
whether enough explicit metadata exists to activate a P12 label contract.

This is **not** a contract activation. It produces a proposal and human-review
packet. Real activation requires a separate human-reviewed PR.

---

## Exact command

```bash
python -m sciencer_d.btc_icft.pipelines.prepare_ds005620_contract_activation \
  --ds-root outputs/btc_icft/ds005620 \
  --out outputs/btc_icft/ds005620_contract_activation
```

With mock fixture (no real data required):

```bash
python -m sciencer_d.btc_icft.pipelines.prepare_ds005620_contract_activation \
  --mock-fixture \
  --ds-root outputs/btc_icft/ds005620 \
  --out outputs/btc_icft/ds005620_contract_activation
```

---

## Expected fixture-safe result

```
[p16] DS005620 contract activation audit complete.
  metadata_file_exists: True
  candidate_label_column: trial_type
  observed_values: ['condition_a', 'condition_b']
  activation_blockers: 1
  contract_activation_allowed: False (always)
  is_ready_for_human_review: False
```

`contract_activation_allowed` is **always false** from this pipeline.

---

## Activation gates

All of the following must be true before a real contract PR can be considered:

| Gate | Meaning |
|---|---|
| `metadata_file_exists` | A local metadata file (events.tsv etc.) was found and is non-empty |
| `explicit_label_column_declared` | A human has declared `explicit_label_column` in the PR |
| `positive_values_declared` | A human has declared `positive_values` |
| `negative_values_declared` | A human has declared `negative_values` |
| `label_scope_declared` | A human has declared `label_scope` |
| `join_keys_declared` | A human has declared `join_keys` |
| `both_classes_present` | Both pos/neg values appear in local metadata |
| `ambiguous_values_rejected` | No `n/a`, `unknown`, `none` etc. remain undeclared |
| `human_review_required` | Always true |
| `contract_activation_allowed` | **Always false** — set only in a separate PR |

---

## Output artifacts

Written to `--out` directory:

| File | Contents |
|---|---|
| `activation_proposal.json` | Gates, candidate column, observed values, blockers |
| `human_review_packet.json` | Reviewer checklist, required declarations, forbidden shortcuts |
| `metadata_value_audit.csv` | Per-value audit: count, is_ambiguous, candidate flags |
| `activation_blockers.json` | Blockers list; `contract_activation_allowed: false` |
| `omega_event.json` | Safe claim and event record |
| `report.md` | Human-readable audit report |

---

## How to inspect outputs

```bash
cat outputs/btc_icft/ds005620_contract_activation/activation_proposal.json
cat outputs/btc_icft/ds005620_contract_activation/activation_blockers.json
cat outputs/btc_icft/ds005620_contract_activation/human_review_packet.json
```

Key fields in `activation_proposal.json`:
- `gates` — all gate values
- `candidate_label_column` — automatically detected column
- `observed_values` — distinct non-ambiguous values found
- `ambiguous_values_found` — values like `n/a`, `unknown` that must be addressed
- `activation_blockers` — human-readable blockers list
- `is_ready_for_human_review` — true only when metadata_file_exists and no blockers

---

## How to activate a real contract

Real contract activation requires a **separate human-reviewed PR** with all of the
following explicitly declared:

1. `explicit_label_column` — e.g., `"trial_type"`
2. `positive_values` — e.g., `["awake"]`
3. `negative_values` — e.g., `["sedated"]`
4. `label_scope` — `"window"`, `"file"`, `"subject"`, or `"session"`
5. `join_keys` — composite key field names for row matching
6. `metadata_provenance` — source of the metadata values
7. `reason_mapping_is_valid` — semantic justification for the mapping
8. `reason_no_shortcut_inference` — explicit statement that no shortcut was used

Do NOT activate a real contract by passing any flag to this pipeline.
This pipeline is permanently read-only.

---

## Guardrails

The following are forbidden in all P16 artifacts:

- proves consciousness
- consciousness proven
- soul proven
- afterlife proven
- liberation detected
- ontology solved
- ultimate reality
- Q equals self / Q equals soul
- Q_abs equals suffering
- f_dress equals karma
- sedated implies no_experience
- unresponsive implies unconscious

Additionally:
- No label inference from file names, topology, or artifacts
- No target fabrication
- No data download
- No modification of P11 promotion gates
- No modification of legacy DS005620 mt_real semantics

---

## Running tests

```bash
python -m pytest tests/btc_icft/test_ds005620_contract_activation.py -q
```

Expected: all tests pass with no failures.

---

## Why this is not contract activation

- P16 only inspects local files — it does not read, parse, or apply any real
  label contract.
- `contract_activation_allowed` is hardcoded to `false` in the pipeline and the
  output artifacts.
- The human-review packet is a checklist for a future human-reviewed PR, not an
  automated trigger.
- No y targets are produced, injected, or passed to any downstream pipeline.
