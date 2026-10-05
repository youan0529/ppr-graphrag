"""Joint output -> source facts, occurrence identities, compatibility graph and directional views."""
import json
from pathlib import Path

import numpy as np

from ppr_graphrag.graph.alignment import canonicalize
from ppr_graphrag.core.text import norm
from ppr_graphrag.graph.identity import review_candidates
from ppr_graphrag.llm.structured_runtime import digest, dump


def prepare(extracted, inverses, docs):
    local, occurrences, facts, identities, issues, dense = {}, {}, [], [], [], []
    sources = {doc['doc_id']: doc for doc in docs}
    for row in extracted:
        did = row['doc_id']
        source = sources[did]
        # Whole-document provenance, supplied by the program, never by the LLM.
        evidence = [dict(doc_id=did, scope='document', start=0, end=len(source['text']), quote=source['text'])]
        for index, original in enumerate(row['facts']):
            fid = 'F_' + digest([did, original])[:24]
            item = dict(original)
            source_kind = item['kind']
            unavailable = source_kind == 'DETAIL_TO_ENTITY' and item['reverse_relation'] is None
            if source_kind == 'TEXT_FACT' or unavailable:
                dense.append(dict(fact_id=fid, doc_id=did, text=item['text'], qualifiers=item['qualifiers'],
                                  source_type=source_kind, route='text', source_fact=original,
                                  fallback_reason='faithful_reverse_unavailable' if unavailable else None,
                                  source_refs=[did], evidence_spans=evidence))
                continue
            reversed_terminal = source_kind == 'DETAIL_TO_ENTITY'
            if reversed_terminal:
                item = dict(item, subject=original['object'], object=original['subject'],
                            relation=original['reverse_relation'], kind='ENTITY_TO_DETAIL')
            ends = {}
            for role in ['subject', 'object']:
                name, typ, disambig = item[role]
                original_role = ('object' if role == 'subject' else 'subject') if reversed_terminal else role
                oid = f'{did}#f{index}:{original_role}'
                is_entity = role == 'subject' or item['kind'] != 'ENTITY_TO_DETAIL'
                lid = did + '#' + digest([norm(name), typ, norm(disambig)])[:20] if is_entity else None
                occurrences[oid] = dict(occurrence_id=oid, doc_id=did, fact_id=fid, role=original_role,
                                        graph_role=role, text=name, type=typ, disambiguator=disambig, local_id=lid)
                ends[role] = (oid, lid)
                if lid:
                    if lid not in local:
                        local[lid] = dict(local_id=lid, doc_id=did, name=name, type=typ,
                                          disambiguator=disambig, description=disambig,
                                          mentions=[], aliases=[], context_facts=[])
                    local[lid]['mentions'].append(dict(name=name, occurrence_id=oid))
            subj, target = ends['subject'][1], ends['object'][1]
            relation = item['relation']
            if source_kind == 'SAME_ENTITY':
                identities.append(dict(fact_id=fid, left=subj, right=target, relation=relation,
                                       basis='explicit_identity', reason='source-extracted identity assertion; not independent verification',
                                       evidence=evidence, source_refs=[did], source_fact=original, qualifiers=item['qualifiers'],
                                       subject_occurrence=ends['subject'][0], object_occurrence=ends['object'][0]))
                continue
            literal = (dict(raw=item['object'][0], value=item['object'][0], type=item['object'][1], normalization='none')
                       if item['kind'] == 'ENTITY_TO_DETAIL' else None)
            inverse = inverses.get(relation, {}) if target else {}
            fact = dict(fact_id=fid, doc_id=did, subject_local=subj, object_local=target,
                        subject_occurrence=ends['subject'][0], object_occurrence=ends['object'][0],
                        relation=f'X {relation} Y', raw_relation=relation, qualifiers=item['qualifiers'], literal=literal,
                        quote=source['text'], evidence_spans=evidence, source_refs=[did], source_fact=original,
                        fact_kind='attribute' if literal else 'relation', source_type=source_kind,
                        source_direction='reverse' if reversed_terminal else 'forward',
                        inverse_relation=inverse.get('inverse'), inverse_reason=inverse.get('reason', 'not applicable' if literal else 'unavailable'))
            facts.append(fact)
            if target and not inverse.get('inverse'):
                # Keep the forward graph fact and add a dense access path for
                # facts whose reverse label is unavailable. Shared fact_id lets
                # later retrieval deduplicate evidence across both access paths.
                text = f"{item['subject'][0]} {relation} {item['object'][0]}"
                if item['qualifiers']:
                    text += ' | ' + json.dumps(item['qualifiers'], ensure_ascii=False)
                dense.append(dict(fact_id=fid, doc_id=did, text=text, qualifiers=item['qualifiers'],
                                  source_type=source_kind, route='reverse_fallback', source_fact=original,
                                  fallback_reason='reverse_label_unavailable',
                                  inverse_status=inverse.get('generation_status', 'unavailable'),
                                  source_refs=[did], evidence_spans=evidence))
            context = f"{item['subject'][0]} | {relation} | {item['object'][0]}"
            if item['qualifiers']:
                context += ' | ' + json.dumps(item['qualifiers'], ensure_ascii=False)
            for lid in [subj, target]:
                if lid:
                    local[lid]['context_facts'].append(context)
    for e in local.values():
        e['context_facts'] = list(dict.fromkeys(e['context_facts']))
    return list(local.values()), occurrences, facts, identities, issues, dense


def build_graph(docs, extracted, inverses, models, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(extractions=extracted, inverses=inverses, alignment=models.config['alignment']))
    if (out / 'graph.json').exists():
        graph = json.loads((out / 'graph.json').read_text())
        if graph['extraction_id'] != identity:
            raise ValueError('Graph input identity changed; use a new output')
        return graph
    local, occurrences, facts, explicit, issues, dense = prepare(extracted, inverses, docs)
    cfg = models.config['alignment']
    names = [norm(e['name']) for e in local]
    contexts = [e['name'] + ' | ' + e['type'] + ' | ' + e['disambiguator'] + ' | ' +
                '; '.join(e['context_facts'][:cfg['context_facts']]) for e in local]
    dump(out / 'status.json', dict(stage='entity_embeddings', local_entities=len(local)))
    name_vec = models.embed(names)
    context_vec = models.embed(contexts)
    np.save(out / 'name_vectors.npy', name_vec)
    np.save(out / 'context_vectors.npy', context_vec)
    np.save(out / 'local_entity_vectors.npy', context_vec)
    dump(out / 'local_entities.json', local)
    dump(out / 'status.json', dict(stage='canonicalization', threshold=cfg['semantic_min']))
    vectors, texts = (name_vec, names) if cfg['representation'] == 'name' else (context_vec, contexts)
    mapping, entities, links, accepted, rejected, stats = canonicalize(local, explicit, vectors, texts, cfg, out,
        review_identity=(lambda pairs, entities: review_candidates(pairs, entities, models, out)) if cfg.get('identity_review') else None)
    views = []
    for fact in facts:
        fact['subject_id'] = mapping[fact['subject_local']]
        fact['object_id'] = mapping[fact['object_local']] if fact['object_local'] else None
        views.append(dict(fact_id=fact['fact_id'], direction='terminal' if fact['literal'] else 'forward',
                          from_local=fact['subject_local'], to_local=fact['object_local'], relation=fact['relation'],
                          source_direction=fact['source_direction']))
        if fact['object_local'] and fact['inverse_relation']:
            views.append(dict(fact_id=fact['fact_id'], direction='reverse', from_local=fact['object_local'],
                              to_local=fact['subject_local'], relation='X ' + fact['inverse_relation'] + ' Y'))
    # Compatibility matrix retains original predicates; separate matrix enables a controlled directional experiment.
    relations = sorted({f['relation'] for f in facts})
    directed = sorted({v['relation'] for v in views})
    dump(out / 'status.json', dict(stage='relation_embeddings', relations=len(relations), directed_relations=len(directed)))
    np.save(out / 'relation_vectors.npy', models.embed(relations))
    np.save(out / 'directional_relation_vectors.npy', models.embed(directed))
    dump(out / 'status.json', dict(stage='dense_fact_embeddings', facts=len(dense)))
    np.save(out / 'dense_fact_vectors.npy', models.embed([f['text'] for f in dense]))
    dump(out / 'dense_facts.json', dense)
    graph = dict(schema='particle-graph-joint-r4-document-sources', extraction_id=identity, entities=entities,
                 local_entities={e['local_id']: e for e in local}, occurrences=occurrences, mapping=mapping,
                 identity_links=links, explicit_identities=accepted, facts=facts, relations=relations,
                 views=views, directional_relations=directed, dense_facts=dense,
                 sources={d['doc_id']: {k: d[k] for k in ['doc_id', 'title', 'text']} for d in docs})
    graph['graph_id'] = digest(graph)
    audit = dict(documents=len(docs), extracted_documents=len(extracted), local_entities=len(local),
                 canonical_entities=len(entities), occurrences=len(occurrences), facts=len(facts),
                 attributes=sum(f['literal'] is not None for f in facts), dense_facts=len(dense),
                 partial_documents=sum(d.get('extraction_status') == 'partial' for d in extracted),
                 reversed_terminals=sum(f['source_direction'] == 'reverse' for f in facts), explicit_identities=len(accepted),
                 dense_reverse_fallbacks=sum(f['route']=='reverse_fallback' for f in dense),
                 inverse_available=sum(bool(f['inverse_relation']) for f in facts),
                 inverse_unavailable=sum(f['literal'] is None and not f['inverse_relation'] for f in facts),
                 alignment=dict(cfg, **stats), rejected_identities=rejected, type_issues=issues,
                 new_threshold_merges=[link for link in links if link.get('basis') == 'embedding' and .8 <= link['score'] < .9],
                 graph_id=graph['graph_id'], source_policy='whole document only; no model-generated evidence refs or paragraph localization',
                 runtime_policy='original-predicate matrix retained for fixed online control; directed views are a separate candidate')
    occurrence_fact_ids = {o['fact_id'] for o in occurrences.values()}
    for f in dense:
        assert f['source_refs'] == [f['doc_id']]
        if f['route'] == 'reverse_fallback':
            assert f['fact_id'] in occurrence_fact_ids
        else:
            assert f['fact_id'] not in occurrence_fact_ids
    for f in facts:
        assert f['subject_local'] in mapping
        assert (f['literal'] is None) == (f['object_local'] is not None)
        for ref in f['evidence_spans']:
            assert graph['sources'][ref['doc_id']]['text'][ref['start']:ref['end']] == ref['quote']
    assert len({f['fact_id'] for f in facts}) == len(facts)
    dump(out / 'graph_report.json', audit)
    dump(out / 'graph.json', graph)
    return graph
