import argparse,json
from pathlib import Path


SOURCE_TO_CONSTRUCT_MAP = {
    "sources": [
        {"source_id": "SRC-TOPOLOGY", "constructs": ["topology_telemetry", "boundary_stability"]},
        {"source_id": "SRC-ACTIVE-INFERENCE", "constructs": ["free_energy_minimization", "allostatic_control"]},
        {"source_id": "SRC-COMP-PSY", "constructs": ["computational_psychiatry_mapping", "symptom_model_interface"]},
        {"source_id": "SRC-BIOELECTRIC", "constructs": ["bioelectric_signal_structures", "basal_cognition_constraints"]},
    ]
}

CONSTRUCT_TO_EQUATION_MAP = {
    "constructs": [
        {"construct": "topology_telemetry", "equation_ids": ["EQ-001", "EQ-002"]},
        {"construct": "free_energy_minimization", "equation_ids": ["EQ-003", "EQ-004"]},
        {"construct": "allostatic_control", "equation_ids": ["EQ-005", "EQ-006"]},
        {"construct": "computational_psychiatry_mapping", "equation_ids": ["EQ-007", "EQ-008"]},
        {"construct": "bioelectric_signal_structures", "equation_ids": ["EQ-009", "EQ-010"]},
    ]
}

CONSTRAINT_TO_FALSIFIER_MAP = {
    "mappings": [
        {"constraint_id": "COS-001", "falsifier_id": "F1"},
        {"constraint_id": "COS-002", "falsifier_id": "F2"},
        {"constraint_id": "GW-001", "falsifier_id": "F3"},
        {"constraint_id": "GW-002", "falsifier_id": "F4"},
        {"constraint_id": "CON-001", "falsifier_id": "F5"},
        {"constraint_id": "CON-002", "falsifier_id": "F6"},
        {"constraint_id": "COS-003", "falsifier_id": "F7"},
        {"constraint_id": "GW-003", "falsifier_id": "F8"},
        {"constraint_id": "CON-003", "falsifier_id": "F9"},
        {"constraint_id": "COS-004", "falsifier_id": "F10"},
    ]
}
def main():
 p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--strict',action='store_true');a=p.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);avail={r:Path(r).exists() for r in a.roots}
 (o/'toe_literature_priority_matrix.json').write_text(json.dumps({'inputs':avail,'status':['not_available' if not v else 'available' for v in avail.values()]},indent=2), encoding='utf-8')
 (o/'toe_theory_integration_digest.md').write_text('P40 integration digest', encoding='utf-8')
 (o/'toe_literature_bridge_report.md').write_text('P40 Summary', encoding='utf-8')
 (o/'generation_manifest.json').write_text(json.dumps({'ok':True},indent=2), encoding='utf-8')
 (o/'source_to_construct_map.json').write_text(json.dumps(SOURCE_TO_CONSTRUCT_MAP,indent=2), encoding='utf-8')
 (o/'construct_to_equation_map.json').write_text(json.dumps(CONSTRUCT_TO_EQUATION_MAP,indent=2), encoding='utf-8')
 (o/'constraint_to_falsifier_map.json').write_text(json.dumps(CONSTRAINT_TO_FALSIFIER_MAP,indent=2), encoding='utf-8')
if __name__=='__main__':main()
