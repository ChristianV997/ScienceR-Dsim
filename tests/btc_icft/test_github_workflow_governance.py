from pathlib import Path


ROOT = Path(".")
PR_TEMPLATE = ROOT / ".github" / "pull_request_template.md"
ISSUE_TEMPLATE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"
PIPELINE_ISSUE = ISSUE_TEMPLATE_DIR / "ds005620_pipeline_task.yml"
RUNTIME_BUG_ISSUE = ISSUE_TEMPLATE_DIR / "science_runtime_bug.yml"
EVIDENCE_CLAIM_ISSUE = ISSUE_TEMPLATE_DIR / "evidence_claim_review.yml"
AGENT_HANDOFF_ISSUE = ISSUE_TEMPLATE_DIR / "agent_handoff.yml"
DOC_AGENT_WORKFLOW = ROOT / "docs" / "github_agent_workflow.md"
DOC_REVIEW_GOVERNANCE = ROOT / "docs" / "ds005620_review_governance.md"
DOC_AGENT_CHECKLISTS = ROOT / "docs" / "agent_pr_checklists.md"

TEMPLATE_FILES = [
    PR_TEMPLATE,
    PIPELINE_ISSUE,
    RUNTIME_BUG_ISSUE,
    EVIDENCE_CLAIM_ISSUE,
    AGENT_HANDOFF_ISSUE,
]

BANNED_PHRASES = [
    "proves consciousness",
    "consciousness proven",
    "soul proven",
    "afterlife proven",
    "liberation detected",
    "ontology solved",
    "ultimate reality",
    "q equals self",
    "q equals soul",
    "q_abs equals suffering",
    "f_dress equals karma",
    "sedated implies no_experience",
    "unresponsive implies unconscious",
    "topology proves liberation",
    "eeg proves consciousness",
]


def test_pr_template_exists():
    assert PR_TEMPLATE.is_file()


def test_pr_template_contains_guardrail_checklist():
    txt = PR_TEMPLATE.read_text(encoding="utf-8")
    assert "## Guardrails Preserved" in txt
    assert "- [ ] No label inference" in txt
    assert "- [ ] No target fabrication" in txt
    assert "- [ ] No empirical claim from mock E2E" in txt


def test_pr_template_contains_tests_run_checklist():
    txt = PR_TEMPLATE.read_text(encoding="utf-8")
    assert "## Tests Run" in txt
    assert "python -m governance.validate" in txt
    assert "make validate-ds005620-e2e-json" in txt


def test_ds005620_pipeline_issue_template_exists():
    assert PIPELINE_ISSUE.is_file()


def test_runtime_bug_issue_template_exists():
    assert RUNTIME_BUG_ISSUE.is_file()


def test_evidence_claim_review_template_exists():
    assert EVIDENCE_CLAIM_ISSUE.is_file()


def test_agent_handoff_template_exists():
    assert AGENT_HANDOFF_ISSUE.is_file()


def test_github_agent_workflow_doc_exists():
    assert DOC_AGENT_WORKFLOW.is_file()


def test_ds005620_review_governance_doc_exists():
    assert DOC_REVIEW_GOVERNANCE.is_file()


def test_agent_pr_checklists_doc_exists():
    assert DOC_AGENT_CHECKLISTS.is_file()


def test_no_template_contains_banned_phrase_substrings():
    for file_path in TEMPLATE_FILES:
        txt = file_path.read_text(encoding="utf-8").lower()
        for phrase in BANNED_PHRASES:
            assert phrase not in txt, f"{file_path} contains banned phrase: {phrase}"


def test_pr_template_mentions_mock_e2e_and_real_local_distinction():
    txt = PR_TEMPLATE.read_text(encoding="utf-8")
    assert "Affects mock E2E only" in txt
    assert "Affects real/local execution" in txt
