"""Human-readable LLM inputs; disk records retain full provenance."""
import json


def chain_text(chain, label, graph=None, reader=False):
    lines = [f'Chain {label}:']
    facts = chain['evidence']['facts']
    if not facts:
        lines.append('Start: ' + str(chain.get('start_entity', 'unknown')) + '; no extension sampled.')
    elif reader:
        for i, f in enumerate(facts, 1):
            lines.append(f"  {i}. {' → '.join(str(x) for x in f['directed_triple'])}")
            if f['qualifiers']:
                lines.append('     Context: ' + '; '.join(f'{k}: {v}' for k,v in f['qualifiers'].items()))
            if f['directed_triple'] != f['source_triple']:
                lines.append('     Original assertion: ' + ' → '.join(str(x) for x in f['source_triple']))
            for link in f['identity_before']:
                lines.append('     Identity connection: ' + ' = '.join(link['names']) +
                    (' (explicit source alias)' if link.get('textual_proof') else ' (unverified embedding merge)'))
    else:
        lines.append('  First assertion source: ' + graph.data['sources'][facts[0]['doc_id']]['title'])
        anchor=chain['evidence'].get('anchor_local')
        if anchor:
            start_name=graph.data['local_entities'][anchor]['name']
            if start_name!=facts[0]['directed_triple'][0]:
                lines.append('  Linked starting name: ' + start_name + ' (graph identity hypothesis)')
        for i, (f, step) in enumerate(zip(facts, chain['path']), 1):
            left,_,right=f['directed_triple']
            if i>1 and facts[i-2]['directed_triple'][2]!=left:
                lines.append('     Linked names: ' + facts[i-2]['directed_triple'][2] + ' ≈ ' + left + ' (graph identity hypothesis)')
            lines.append(f"  {i}. {left} → {f['directed_triple'][1]} → {right}" +
                (' [terminal detail]' if step['to_node'] is None else ''))
    if not reader:
        needs = list(dict.fromkeys(p['question'] for p in chain.get('local_parts', [])))
        if needs:
            lines.append('Saved contribution: ' + '; '.join(needs))
    return '\n'.join(lines)


def reader_text(question, chains, documents):
    lines = ['Question: ' + question, '', 'Candidate evidence:']
    lines += [chain_text(c, c.get('id',c.get('path_id')), reader=True) for c in chains]
    lines += ['', 'Source articles:']
    for i, doc in enumerate(documents, 1):
        if isinstance(doc, dict):
            lines += [f"Article {i}: {doc.get('title',doc.get('doc_id',''))}",
                      doc.get('text',doc.get('content',json.dumps(doc,ensure_ascii=False)))]
        else:
            lines += [f'Article {i}:', str(doc)]
    return '\n\n'.join(lines)
