import argparse,json
from pathlib import Path

def _load_list(path: Path):
    if not path.exists():
        return []
    try:
        d=json.loads(path.read_text())
    except Exception:
        return []
    return d if isinstance(d,list) else []

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
 root=Path(a.root)
 eq=[x.get('equation_id') for x in _load_list(root/'equation_candidate_registry.json')]
 ff=[x.get('id') for x in _load_list(root/'toe_falsifier_watchlist.json')]
 (o/'rag_safe_documents.json').write_text(json.dumps({'safe':['bridge'],'registry_summary':{'equation_ids':eq,'falsifier_ids':ff}},indent=2))
 (o/'rag_chunking_plan.json').write_text(json.dumps({'chunks':['registry_summary','constraint_matrices','guardrails']},indent=2))
 (o/'rag_query_examples.json').write_text(json.dumps({'examples':['List equation candidates EQ-001..EQ-010 with claim-scope caveats.','Does this validate the TOE? No.','Can Qabs prove consciousness? No.']},indent=2))
 (o/'rag_forbidden_answer_patterns.json').write_text(json.dumps(['TOE validated','final theory','consciousness solved','Q proves','Qabs proves','clinical treatment','diagnosis','soul proof','afterlife proof','ontology promotion'],indent=2))
 (o/'rag_public_answer_guidelines.md').write_text('Do not claim validation. Keep registry outputs as hypothesis scaffolding only.')
 (o/'rag_pack_report.md').write_text('RAG pack report with populated TOE registry summaries.')
if __name__=='__main__':main()
