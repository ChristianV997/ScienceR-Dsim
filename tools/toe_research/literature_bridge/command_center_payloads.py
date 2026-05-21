import argparse,json
from pathlib import Path
from datetime import datetime,timezone

FILES=['toe_literature_bridge_status.json','toe_priority_matrix.json','toe_topology_telemetry_upgrade.json','toe_cosmology_constraints.json','toe_gravitational_wave_constraints.json','toe_consciousness_adversarial_matrix.json','toe_equation_registry.json','toe_falsifier_watchlist.json']

def _load_list(path: Path):
    if not path.exists():
        return []
    try:
        data=json.loads(path.read_text())
    except Exception:
        return []
    return data if isinstance(data,list) else []

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
 root=Path(a.root)
 equations=_load_list(root/'equation_candidate_registry.json')
 cosmology=_load_list(root/'cosmology_constraint_matrix.json')
 gw=_load_list(root/'gravitational_wave_constraint_matrix.json')
 consciousness=_load_list(root/'consciousness_theory_adversarial_matrix.json')
 falsifiers=_load_list(root/'toe_falsifier_watchlist.json')
 base={'ok':True,'generated_at':datetime.now(timezone.utc).isoformat(),'toe_validation_claims_allowed':False,'final_theory_claims_allowed':False,'consciousness_proof_claims_allowed':False,'clinical_claims_allowed':False,'ontology_promotion_allowed':False,'q_qabs_fdress_proof_claims_allowed':False,'real_data_auto_execution_allowed':False,'live_api_calls_allowed':False,'api_key_exposure_allowed':False,'active_goal':'P40.2','preset':'toe_research','required_surfaces':['package'],'guardrails':['no proof'],'forbidden_patterns':['final theory'],'next_actions':['validate'],'registry_summary':{'equations':[e.get('equation_id') for e in equations],'cosmology_constraints':[c.get('constraint_id') for c in cosmology],'gw_constraints':[c.get('constraint_id') for c in gw],'consciousness_theories':[c.get('theory') for c in consciousness],'falsifiers':[f.get('id') for f in falsifiers]}}
 for f in FILES:(o/f).write_text(json.dumps(base,indent=2))
if __name__=='__main__':main()
