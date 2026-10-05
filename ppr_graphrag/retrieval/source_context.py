"""Optional primary articles for actual named chain participants, within Reader budget."""
from ppr_graphrag.core.text import norm
from ppr_graphrag.retrieval.chain_display import reader_text


def add_context(result,graph,models,config,reader_prompt):
    required=list(result['supporting_doc_ids'])
    result['chain_source_doc_ids']=required
    result['additional_context_doc_ids']=[]
    if not config['reader'].get('participant_articles',False) or not result['selected_paths']:
        return
    by_title={norm(d['title']):doc_id for doc_id,d in graph['sources'].items()}
    names=[]
    for chain in result['selected_paths']:
        for fact in chain['evidence']['facts']:
            names.append(fact['source_triple'][0])
            if fact['literal'] is None:names.append(fact['source_triple'][2])
    candidates=list(dict.fromkeys(by_title[norm(n)] for n in names if norm(n) in by_title))
    chosen=list(required)
    for doc_id in candidates:
        if doc_id in chosen or len(chosen)>=config['reader']['top_k']:continue
        attempt=chosen+[doc_id]
        text=reader_text(result['original_question'],result['selected_paths'],[graph['sources'][d] for d in attempt])
        messages=[dict(role='system',content=reader_prompt),dict(role='user',content=text)]
        if models.token_count(messages)+config['reader']['output_tokens']<=config['llm']['context']:
            chosen=attempt
    result['supporting_doc_ids']=chosen
    result['additional_context_doc_ids']=[d for d in chosen if d not in required]
