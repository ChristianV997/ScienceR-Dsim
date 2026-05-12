# DS005620 Activation Declaration Validator Runbook (P17.0)

---

## Purpose

P17.0 validates a human-authored DS005620 activation declaration for
completeness, provenance, and no-shortcut safeguards before any real
contract activation.

P17.0 produces a dry-run preview but does NOT activate any P12 contract.
Real activation requires a separate P17.1 contract activation PR.

---

## Relation to P16

P16 produced:
- `activation_proposal.json` — candidate columns, unresolved values, blockers
- `human_review_packet.json` — required decisions checklist

P17.0 takes the output of P16 (the activation packet) and a human-authored
declaration JSON, validates the declaration, and produces:
- `activation_declaration_validation.json` — pass/fail, errors, warnings
- `activation_contract_preview.json` — P12-compatible preview (not active)
- `activation_declaration_errors.json` — errors and blockers
- `activation_declaration_template.json` — fillable template
- `omega_event.json` — safe claim record
- `report.md` — human-readable validation report

---

## What P17.0 Does

- Validates all required declaration fields for completeness
- Checks `no_shortcut_inference_confirmation` for required phrases
- Checks `metadata_provenance` is not `"unknown"`
- Checks `semantic_justification` is at least 40 characters
- Cross-checks declared values against P16 `unresolved_values` (warnings only)
- Produces a `activation_contract_preview.json` with status `preview_human_reviewed_not_active`
- Sets `activation_dry_run_allowed=true` only if `declaration_valid=true`

---

## What P17.0 Does NOT Do

- Does **not** activate any P12 contract (`real_contract_activation_allowed` is always `false`)
- Does **not** infer labels from file names, topology, or artifacts
- Does **not** fabricate y targets
- Does **not** run P11 target-aware benchmarking
- Does **not** modify P11/P12/P13 behavior
- Does **not** modify legacy DS005620 `mt_real` semantics
- Does **not** download data
- Does **not** modify any P12 source file

---

## Declaration Schema

The human-authored declaration JSON must include all of the following:

| Field | Type | Requirement |
|---|---|---|
| `dataset_id` | `str` | Must be `"DS005620"` |
| `explicit_label_column` | `str` | Non-empty column name from metadata |
| `positive_values` | `list[str]` | Non-empty; must not overlap with negative_values |
| `negative_values` | `list[str]` | Non-empty; must not overlap with positive_values |
| `label_scope` | `str` | One of: `window`, `file`, `run`, `subject`, `session` |
| `join_keys` | `list[str]` | Must include all 8 strict window keys |
| `metadata_provenance` | `str` | Non-empty; not `"unknown"` |
| `semantic_justification` | `str` | At least 40 characters |
| `no_shortcut_inference_confirmation` | `str` | Must include both required phrases |
| `reviewer_identity_or_role` | `str` | Non-empty |
| `review_date` | `str` | Non-empty |
| `both_classes_present_confirmation` | `bool` | Must be `true` |
| `ambiguity_reviewed` | `bool` | Must be `true` |

Optional:
- `source_activation_packet` — path to P16 `activation_proposal.json`
- `notes` — list of strings

Required `no_shortcut_inference_confirmation` phrases:
- `"no sedated-to-no_experience shortcut"`
- `"no unresponsive-to-unconscious shortcut"`

Required join_keys (all 8):
`dataset_id`, `row_id`, `source_file`, `window_id`, `window_start_s`,
`window_end_s`, `sample_start`, `sample_end`

---

## Exact Template Command

```bash
python -m sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration \
  --write-template \
  --out outputs/btc_icft/ds005620_activation_declaration
```

---

## Exact Mock-Valid Command

```bash
python -m sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration \
  --mock-valid-declaration \
  --out outputs/btc_icft/ds005620_activation_declaration
```

Expected result:
```
declaration_valid: True
activation_dry_run_allowed: True
real_contract_activation_allowed: False
validation_errors: 0
```

---

## Exact Real/Local Declaration Command

```bash
python -m sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration \
  --declaration data/DS005620/ds005620_activation_declaration.json \
  --activation-packet outputs/btc_icft/ds005620_contract_activation/activation_proposal.json \
  --out outputs/btc_icft/ds005620_activation_declaration
```

---

## Validation Failure Modes

| Failure | Cause |
|---|---|
| `missing_required_field:X` | Required field not in declaration JSON |
| `invalid_dataset_id` | dataset_id is not `"DS005620"` |
| `explicit_label_column must be a non-empty string` | Empty or missing column |
| `positive_values must be a non-empty list` | No positive values declared |
| `negative_values must be a non-empty list` | No negative values declared |
| `positive_values and negative_values must not overlap` | Values appear in both |
| `unsupported label_scope` | Not one of the valid scopes |
| `join_keys missing required strict keys` | Missing one of the 8 required keys |
| `metadata_provenance must not be 'unknown'` | Provenance is `"unknown"` |
| `semantic_justification must be at least 40 characters` | Too short |
| `no_shortcut_inference_confirmation must explicitly include: X` | Missing phrase |
| `both_classes_present_confirmation must be true` | Set to `false` |
| `ambiguity_reviewed must be true` | Set to `false` |

---

## Next Step: P17.1 Contract Activation PR

After P17.0 validation passes (`declaration_valid: True`), open a
**separate P17.1 contract activation PR** that:

1. Includes the validated declaration JSON
2. Runs P12 → P13 → P11 target-aware benchmark
3. Has been independently peer-reviewed

Do NOT merge P17.1 PR without peer review.
Do NOT open P17.1 PR if `declaration_valid: False`.

---

## Guardrails

All P17.0 artifacts are scanned for forbidden phrases:

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
- `real_contract_activation_allowed` is hardcoded `false` everywhere
- `activation_contract_preview` has status `preview_human_reviewed_not_active`
- Preview does not modify any P12 source file
