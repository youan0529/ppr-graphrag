"""Only explicit directed views are walked; identity remains a connection annotation."""
from collections import defaultdict
import heapq
import itertools

import numpy as np

from ppr_graphrag.llm.structured_runtime import digest


class FactGraph:
    def __init__(self, data, relation_vectors, name_vectors=None):
        self.data = data
        self.facts = {f['fact_id']: f for f in data['facts']}
        self.relations = data['directional_relations']
        self.relation_vectors = np.asarray(relation_vectors, dtype=np.float32)
        self.name_vectors = name_vectors
        self.local_index = {lid: i for i, lid in enumerate(data['local_entities'])}
        if self.relation_vectors.shape != (len(self.relations), 4096):
            raise ValueError('Directional relation vector order/shape mismatch')
        if name_vectors is not None and name_vectors.shape != (len(self.local_index), 4096):
            raise ValueError('Local name vector shape mismatch')
        self.adjacency, self.identity = defaultdict(list), defaultdict(list)
        seen = set()
        for view in data['views']:
            fact = self.facts[view['fact_id']]
            if view['direction'] not in ('forward', 'reverse', 'terminal'):
                raise ValueError('Unknown directed view')
            if fact['literal'] is not None:
                if (view['direction'] != 'terminal' or view['to_local'] is not None
                        or fact.get('inverse_relation') is not None):
                    raise ValueError('A terminal detail must end the path and cannot have a reverse edge')
            elif view['direction'] == 'terminal':
                raise ValueError('An entity-to-entity view cannot be marked terminal')
            forward = view['direction'] != 'reverse'
            expected_from = fact['subject_local'] if forward else fact['object_local']
            expected_to = fact['object_local'] if forward else fact['subject_local']
            expected_relation = fact['relation'] if forward else 'X ' + (fact.get('inverse_relation') or '') + ' Y'
            if view['from_local'] != expected_from or view['to_local'] != expected_to or view['relation'] != expected_relation:
                raise ValueError('View differs from its source fact/recorded inverse')
            if not forward and (fact['literal'] is not None or not fact.get('inverse_relation')):
                raise ValueError('Cannot reverse an attribute or invent an inverse')
            step = dict(view, from_node=data['mapping'][view['from_local']],
                        to_node=data['mapping'][view['to_local']] if view['to_local'] else None)
            key = digest(step)
            if key not in seen:
                seen.add(key)
                self.adjacency[step['from_node']].append(step)
        # Include all accepted explicit assertions, not just a historical spanning tree.
        seen = set()
        for link in data['identity_links'] + data.get('explicit_identities', []):
            key = digest(link)
            if key not in seen:
                seen.add(key)
                self.identity[link['left']].append((link['right'], link))
                self.identity[link['right']].append((link['left'], link))
        self.access_report = dict(facts=len(self.facts), views=sum(map(len, self.adjacency.values())),
            forward_views=sum(v['direction'] == 'forward' for v in data['views']),
            reverse_views=sum(v['direction'] == 'reverse' for v in data['views']),
            terminal_views=sum(v['direction'] == 'terminal' for v in data['views']),
            missing_inverse_facts=sum(f['object_local'] is not None and not f.get('inverse_relation') for f in data['facts']),
            policy='explicit directed views only; no implicit incoming-edge fallback')

    def endpoint_local(self, step, at_from):
        return step['from_local' if at_from else 'to_local']

    def identity_evidence(self, left, right, fact_docs):
        if left == right:
            return []
        if self.data['mapping'][left] != self.data['mapping'][right]:
            raise ValueError('Identity connection crosses canonical entities')
        # Prefer explicit textual assertions, then lower additive source cost and fewer links.
        # This is a route heuristic, not a claim of globally minimal sufficient provenance.
        queue = [((0, 0, 0), 0, left, [])]
        serial = itertools.count(1)
        best = {left: (0, 0, 0)}
        while queue:
            cost, _, current, route = heapq.heappop(queue)
            if cost != best[current]:
                continue
            if current == right:
                return route
            neighbors = list(self.identity[current])
            for target, link in neighbors:
                added = (int(link.get('basis') != 'explicit_identity'),
                         len({e['doc_id'] for e in link['evidence']} - fact_docs), 1)
                candidate = tuple(a+b for a, b in zip(cost, added))
                if target not in best or candidate < best[target]:
                    best[target] = candidate
                    heapq.heappush(queue, (candidate, next(serial), target, route + [link]))
        raise ValueError(f'Missing identity provenance between {left} and {right}')

    def validate_path(self, path, start=None):
        current, used = start, set()
        for i, step in enumerate(path):
            if current is None:
                current = step['from_node']
            if step['from_node'] != current or step not in self.adjacency[current]:
                raise ValueError('Invalid or disconnected directed view')
            if step['fact_id'] in used:
                raise ValueError('Repeated source fact, including its reverse view')
            used.add(step['fact_id'])
            current = step['to_node']
            if current is None and i != len(path)-1:
                raise ValueError('Attribute must terminate the chain')
        return current

    def evidence(self, path, anchor_local=None):
        self.validate_path(path)
        facts, dependencies = [], []
        previous = anchor_local
        fact_docs = {self.facts[s['fact_id']]['doc_id'] for s in path}
        for step in path:
            f = self.facts[step['fact_id']]
            left, right = step['from_local'], step['to_local']
            links = self.identity_evidence(previous, left, fact_docs) if previous and previous != left else []
            annotated = [dict(d, names=[self.data['local_entities'][d[k]]['name'] for k in ['left', 'right']],
                              textual_proof=d.get('basis') == 'explicit_identity') for d in links]
            dependencies += annotated
            subject = self.data['local_entities'][f['subject_local']]['name']
            obj = self.data['local_entities'][f['object_local']]['name'] if f['object_local'] else None
            from_name = self.data['local_entities'][left]['name']
            to_name = self.data['local_entities'][right]['name'] if right else f['literal']['raw']
            predicate = f['raw_relation'] if step['direction'] != 'reverse' else f['inverse_relation']
            facts.append(dict(fact_id=f['fact_id'], subject=subject, object=obj, literal=f['literal'],
                relation=f['relation'], raw_relation=f['raw_relation'], qualifiers=f['qualifiers'],
                doc_id=f['doc_id'], quote=f['quote'], evidence_spans=f.get('evidence_spans', []), visit=step,
                directed_triple=[from_name, predicate, to_name],
                source_triple=[subject, f['raw_relation'], obj if obj is not None else f['literal']['raw']],
                identity_before=annotated,
                traversal=dict(from_name=from_name, to_name=to_name, direction=step['direction'])))
            facts[-1]['traversal'].update({'from': from_name, 'to': to_name})
            previous = right
        dependencies = list({digest(d): d for d in dependencies}.values())
        docs = list(dict.fromkeys([f['doc_id'] for f in facts] +
            [e['doc_id'] for d in dependencies if d.get('textual_proof') for e in d['evidence']]))
        # Similarity provenance records why a connection was proposed, not a textual proof of identity.
        hypothesis_docs = list(dict.fromkeys(e['doc_id'] for d in dependencies
            if not d.get('textual_proof') for e in d['evidence']))
        return dict(facts=facts, identity_support=dependencies, doc_ids=docs,
                    identity_hypothesis_doc_ids=hypothesis_docs, anchor_local=anchor_local)

    def classify(self, hints, models, threshold):
        hints = sorted(set(hints))
        if not hints:
            return set(), {'hints': [], 'scores': {r: None for r in self.relations}}
        vectors = models.embed(hints)
        values = (self.relation_vectors @ vectors.T).max(axis=1)
        scores = {r: float(v) for r, v in zip(self.relations, values)}
        return {r for r, v in scores.items() if v >= threshold}, dict(hints=hints, scores=scores)

    def probabilities(self, node, path, priority, epsilon):
        edges = self.adjacency[node]
        flags = [e['relation'] in priority for e in edges]
        np_, no = sum(flags), len(edges)-sum(flags)
        if np_ and no:
            weights = [(1-epsilon)/np_ if flag else epsilon/no for flag in flags]
        elif np_:
            weights = [1/np_] * np_
        elif no:
            weights = [1/no if epsilon > 0 else 0] * no
        else:
            return [], []
        used = {s['fact_id'] for s in path}
        weights = [w if e['fact_id'] not in used else 0 for e, w in zip(edges, weights)]
        total = sum(weights)
        return edges, [w/total if total else 0 for w in weights]
