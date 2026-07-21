# DS005620 Reviewed Contract Activation Runbook (P17.1)

---

## Purpose

P17.1 materializes a P12-compatible reviewed external contract artifact from
a valid P17.0 human-authored activation declaration.

P17.1 does NOT modify any P12 source file.
P17.1 does NOT infer labels or emit y targets.
P17.1 does NOT run P11 benchmarking.

`activation_allowed` is `true` in the output artifact ONLY when the
declaration is valid. A separate P18 PR (with peer review) is required
before running P12 → P13 → P11.

---

## Relation to P17.0

P17.0 produced:
- `activation_declaration_validation.json` — pass/fail, errors, warnings
- `activation_contract_preview.json` — P12-compatible preview (not active)

P17.1 takes:
- A human-authored declaration JSON (same format as P17.0 input)
- Optionally: the P17.0 `activation_declaration_validation.json` for cross-check

P17.1 produces:
- `reviewed_contract.json` — reviewed external contract record
- `p12_external_contract.json` — P12-compatible active contract artifact
- `p18_handoff.json` — P18 readiness and recommended commands
- `reviewed_activation_report.json` — validation and guardrail flags
- `omega_event.json` — safe claim record
- `report.md` — human-readable materialization report

---

## What P17.1 Does

- Re-validates the human-authored declaration (same rules as P17.0)
- Materializes a `p12_external_contract.json` with P12-compatible field names
  only when `reviewed_contract_valid=true`
- Sets `activation_allowed=true` in the output artifact only for valid declarations
- Sets all P18 readiness flags only for valid declarations
- Emits an omega event with `source_contracts_modified=false` and `p11_run=false`

---

## What P17.1 Does NOT Do

- Does **not** modify any P12 source file
- Does **not** infer labels from file names, topology, or artifacts
- Does **not** fabricate y targets
- Does **not** run P11 target-aware benchmarking
- Does **not** modify P11/P12/P13 behavior
- Does **not** modify legacy DS005620 `mt_real` semantics
- Does **not** download data
- Does **not** add new dependencies

---

## Contract Status Values

| Condition | `contract_status` in `p12_external_contract.json` |
|---|---|
| Valid declaration | `active_reviewed_external_contract` |
| Invalid declaration | `blocked_invalid_declaration` |

---

## Exact Mock-Valid Command

```bash
python -m sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract \
  --mock-valid-declaration \
  --out outputs/btc_icft/ds005620_reviewed_contract
```

Expected result:
```
reviewed_contract_valid: True
activation_allowed: True
activation_errors: 0
ready_for_p12_alignment: True
```

---

## Exact Mock-Invalid Command

```bash
python -m sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract \
  --mock-invalid-declaration \
  --out outputs/btc_icft/ds005620_reviewed_contract_invalid
```

Expected result:
```
reviewed_contract_valid: False
activation_allowed: False
activation_errors: > 0
ready_for_p12_alignment: False
```

---

## Exact Real/Local Declaration Command

```bash
python -m sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract \
  --declaration data/DS005620/ds005620_activation_declaration.json \
  --validation outputs/btc_icft/ds005620_activation_declaration/activation_declaration_validation.json \
  --out outputs/btc_icft/ds005620_reviewed_contract
```

The `--validation` flag is optional. If provided, P17.0 validation warnings
are propagated into the P17.1 output. The declaration is re-validated
independently regardless.

---

## Declaration Schema

Same as P17.0 (see `ds005620_activation_declaration_runbook.md`).

All 13 required fields must be present and valid:

| Field | Requirement |
|---|---|
| `dataset_id` | Must be `"DS005620"` |
| `explicit_label_column` | Non-empty |
| `positive_values` | Non-empty; no overlap with negative |
| `negative_values` | Non-empty; no overlap with positive |
| `label_scope` | One of: `window`, `file`, `run`, `subject`, `session` |
| `join_keys` | Must include all 8 strict window keys |
| `metadata_provenance` | Non-empty; not `"unknown"` |
| `semantic_justification` | At least 40 characters |
| `no_shortcut_inference_confirmation` | Must include both required phrases |
| `reviewer_identity_or_role` | Non-empty |
| `review_date` | Non-empty |
| `both_classes_present_confirmation` | Must be `true` |
| `ambiguity_reviewed` | Must be `true` |

---

## P12 External Contract Fields

`p12_external_contract.json` uses P12-compatible field names:

| Field | Source |
|---|---|
| `dataset_id` | From declaration |
| `contract_status` | `active_reviewed_external_contract` (valid only) |
| `explicit_label_column` | From declaration |
| `positive_values` | From declaration |
| `negative_values` | From declaration |
| `label_scope` | From declaration |
| `join_keys` | From declaration |
| `metadata_provenance` | From declaration |
| `activation_provenance` | `p17_1_reviewed_materializer` |
| `guardrails` | P17.1 guardrail list |

---

## P18 Handoff

`p18_handoff.json` indicates downstream readiness:

| Flag | Condition |
|---|---|
| `ready_for_p12_alignment` | `reviewed_contract_valid=true` |
| `ready_for_p13_target_injection` | `reviewed_contract_valid=true` |
| `ready_for_p11_target_aware_benchmark` | `reviewed_contract_valid=true` |

P18 also requires external inputs that P17.1 does not provide:
- Real/local DS005620 metadata file (events.tsv or equivalent)
- DS005620 signal windows or MNE extraction outputs (P19.1)
- P12 alignment output, P13 labeled features, P10 topology features
- P11 target-aware benchmark output

---

## Guardrails

All P17.1 artifacts are scanned for forbidden phrases:

- proves consciousness / consciousness proven
- soul proven / afterlife proven
- liberation detected / ontology solved
- ultimate reality
- Q equals self / Q equals soul
- Q_abs equals suffering / f_dress equals karma
- sedated implies no_experience
- unresponsive implies unconscious
- topology proves liberation / EEG proves consciousness

Additionally:
- `source_contracts_modified` is hardcoded `false` in omega_event
- `p11_run` is hardcoded `false` in omega_event
- `reviewed_activation_report.json` always includes:
  - `no_label_inference: true`
  - `no_target_fabrication: true`
  - `no_source_contract_modified: true`
  - `no_p11_run: true`

---

## Next Step: P18 PR

After P17.1 produces a valid reviewed external contract:

1. Peer-review `p12_external_contract.json` with an independent reviewer.
2. Open a **P18 PR** that includes:
   - The reviewed external contract JSON
   - P12 alignment run with `--external-contract` flag
   - P13 target injection
   - P11 target-aware benchmark
   - All benchmark artifacts and metrics
3. Do NOT merge P18 PR without peer review.
4. Do NOT open P18 PR if `reviewed_contract_valid: False`.
