"""Occurrence-aware graph adapter; complete-link alignment inherited from r2b/r9."""
import json
from ppr_graphrag.core.text import norm
from collections import defaultdict
from pathlib import Path
import faiss
import numpy as np
from ppr_graphrag.llm.structured_runtime import digest




def type_group(kind):
    return "OTHER" if kind in ("Other", "Unknown") else kind


def compatible(left, right, explicit=False):
    return type_group(left) == type_group(right) or (explicit and "OTHER" in (type_group(left), type_group(right)))


def local_conflict(left, right, local):
    seen = {(local[i]["doc_id"], norm(local[i]["name"])): local[i]["local_id"] for i in left}
    return any((local[j]["doc_id"], norm(local[j]["name"])) in seen and
               seen[local[j]["doc_id"], norm(local[j]["name"])] != local[j]["local_id"] for j in right)

def canonicalize(local, explicit, vectors, texts, cfg, out, review_identity=None):
    """Explicit components first; complete-link between these components prevents similarity chains.

    An explicit alias component can contain dissimilar names. Its similarity to another component
    is the maximum over its supported names; complete-link is applied BETWEEN these components.
    """
    n = len(local)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}
    index = {e["local_id"]: i for i, e in enumerate(local)}
    links, rejected = [], []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j, link):
        a, b = root(i), root(j)
        if a == b:
            return
        if (len(members[a]), -a) < (len(members[b]), -b):
            a, b = b, a
        parent[b] = a
        members[a].extend(members.pop(b))
        links.append(link)

    accepted_explicit = []
    for link in explicit:
        i, j = index[link["left"]], index[link["right"]]
        left_types = {local[k]["type"] for k in members[root(i)]}
        right_types = {local[k]["type"] for k in members[root(j)]}
        if all(compatible(a, b, True) for a in left_types for b in right_types):
            accepted_explicit.append(link)
            join(i, j, link)
        else:
            rejected.append(dict(link, rejection="explicit_identity_type_conflict"))
    # Atomic components are formed solely from explicit identity evidence.
    atoms = {r: list(ids) for r, ids in members.items()}
    cluster_atoms = {r: {r} for r in members}
    # Each explicit alias component is one article-local identity group. Once
    # those groups differ within an article, vector merging must not reunite
    # them, including indirectly through a different article.
    article_groups = {r: {local[i]['doc_id']: r for i in ids} for r, ids in atoms.items()}

    def article_conflict(a, b):
        left, right = article_groups[a], article_groups[b]
        return any(doc in right and group != right[doc] for doc, group in left.items())

    # Deduplicate equal representations within explicit components, with no identity assumption.
    atom_signatures = {r: tuple(sorted({texts[i] for i in ids})) for r, ids in atoms.items()}
    atom_vectors = {}
    for r, ids in atoms.items():
        unique = {texts[i]: i for i in ids}
        atom_vectors[r] = vectors[list(unique.values())]
    groups = defaultdict(list)
    for i, e in enumerate(local):
        groups[type_group(e["type"])].append(i)
    candidates = {}
    faiss.omp_set_num_threads(cfg.get("threads", 4))
    threshold = cfg["semantic_min"]
    for group, ids in groups.items():
        mat = np.ascontiguousarray(vectors[ids], dtype=np.float32)
        idx = faiss.IndexFlatIP(vectors.shape[1])
        idx.add(mat)
        k = min(len(ids), cfg["candidate_k"] + 1)
        for start in range(0, len(ids), 256):
            scores, near = idx.search(mat[start : start + 256], k)
            for off, (ss, nn) in enumerate(zip(scores, near)):
                i = ids[start + off]
                for score, j0 in zip(ss, nn):
                    j = ids[int(j0)]
                    if i != j and score >= threshold:
                        pair = tuple(sorted((i, j)))
                        candidates[pair] = float(score)
        # Exact-name buckets avoid losing common-name occurrences to top-k ties. In context mode,
        # name equality is ONLY a candidate route and must still pass the vector threshold.
        buckets = defaultdict(list)
        for i in ids:
            buckets[norm(local[i]["name"])].append(i)
        for bucket in buckets.values():
            for j in bucket[1:]:
                i = bucket[0]
                score = float(vectors[i] @ vectors[j])
                if score >= threshold:
                    candidates[tuple(sorted((i, j)))] = score

    # OTHER is unknown, not a type conflict. Retrieve across the unknown/known
    # boundary in both directions; retain the existing threshold and cluster checks.
    unknown = groups.get("OTHER", [])
    known = [i for group, ids in groups.items() if group != "OTHER" for i in ids]
    for source_ids, target_ids in ((unknown, known), (known, unknown)):
        if not source_ids or not target_ids:
            continue
        idx = faiss.IndexFlatIP(vectors.shape[1])
        idx.add(np.ascontiguousarray(vectors[target_ids], dtype=np.float32))
        k = min(len(target_ids), cfg["candidate_k"])
        for start in range(0, len(source_ids), 256):
            batch_ids = source_ids[start:start + 256]
            scores, near = idx.search(np.ascontiguousarray(vectors[batch_ids], dtype=np.float32), k)
            for i, ss, nn in zip(batch_ids, scores, near):
                for score, j0 in zip(ss, nn):
                    if score >= threshold:
                        candidates[tuple(sorted((i, target_ids[int(j0)])))] = float(score)
    # Include exact-name cross-type candidates even when top-k ties are crowded.
    known_names = defaultdict(list)
    for i in known:
        known_names[norm(local[i]["name"])].append(i)
    for i in unknown:
        for j in known_names[norm(local[i]["name"])]:
            score = float(vectors[i] @ vectors[j])
            if score >= threshold:
                candidates[tuple(sorted((i, j)))] = score

    # Explicit aliases retain their source assertion. Equal normalized names
    # retain the previous name-matching policy; different names need identity
    # review before a vector candidate can become a merge.
    possible_review = [(i, j) for i, j in sorted(candidates) if root(i) != root(j)
                       and norm(local[i]['name']) != norm(local[j]['name'])
                       and compatible(local[i]['type'], local[j]['type'], True)]
    review_pairs = [(i, j) for i, j in possible_review
                    if local[i]['doc_id'] != local[j]['doc_id']
                    and not article_conflict(root(i), root(j))]
    filter_stats = dict(review_candidates_before_document_filter=len(possible_review),
                        review_candidates_after_document_filter=len(review_pairs),
                        document_filtered_before_review=len(possible_review)-len(review_pairs),
                        base_batches_before=(len(possible_review)+3)//4,
                        base_batches_after=(len(review_pairs)+3)//4)
    (Path(out)/'identity_candidate_filter.json').write_text(json.dumps(filter_stats, indent=2))
    identity_decisions = review_identity(review_pairs, local) if review_identity else {}
    different = defaultdict(set)
    for (i, j), result in identity_decisions.items():
        if result['decision'] == 'different':
            different[i].add(j)
            different[j].add(i)

    def coherent(a, b):
        # Repeated identical signatures need only one comparison, important for frequent names.
        aa = {atom_signatures[x]: x for x in cluster_atoms[a]}
        bb = {atom_signatures[x]: x for x in cluster_atoms[b]}
        for sa, x in aa.items():
            for sb, y in bb.items():
                if sa != sb and float(np.max(atom_vectors[x] @ atom_vectors[y].T)) < threshold:
                    return False
        return True

    counts = defaultdict(int)
    decision_records = []
    for (i, j), score in sorted(candidates.items(), key=lambda x: (-x[1], x[0])):
        a, b = root(i), root(j)
        if a == b:
            counts["already_joined"] += 1
            continue
        # OTHER must not bridge incompatible explicit types transitively.
        types_a = {local[x]["type"] for x in members[a]}
        types_b = {local[x]["type"] for x in members[b]}
        decision = "merge"
        if not all(compatible(x, y, True) for x in types_a for y in types_b):
            decision = "type_conflict"
        elif article_conflict(a, b):
            decision = 'same_document_distinct_groups'
        elif local_conflict(members[a], members[b], local):
            decision = "document_homonym_conflict"
        elif cfg["clustering"] == "complete_link" and not coherent(a, b):
            decision = "cluster_inconsistent"
        identity_result = identity_decisions.get((i, j))
        if decision == 'merge' and review_identity:
            if norm(local[i]['name']) != norm(local[j]['name']) and (not identity_result or identity_result['decision'] != 'same'):
                decision = 'identity_different' if identity_result and identity_result['decision'] == 'different' else 'identity_unknown'
            elif any(different[x].intersection(members[b]) for x in members[a]):
                decision = 'identity_cluster_conflict'
        counts[decision] += 1
        record = {
            "left": local[i]["local_id"],
            "right": local[j]["local_id"],
            "score": score,
            "representation": cfg["representation"],
            "decision": decision,
        }
        if identity_result:
            record['identity_review'] = identity_result
        decision_records.append(record)
        if decision == "merge":
            link = dict(
                record,
                basis="embedding",
                reason="identity-reviewed vector candidate; not textual identity proof" if identity_result else "equal-name/type-compatible vector merge; not textual identity proof",
                evidence=[{"doc_id": d} for d in sorted({local[i]["doc_id"], local[j]["doc_id"]})],
            )
            combined = cluster_atoms[a] | cluster_atoms[b]
            combined_articles = {**article_groups[a], **article_groups[b]}
            join(i, j, link)
            article_groups.pop(a)
            article_groups.pop(b)
            article_groups[root(i)] = combined_articles
            cluster_atoms.pop(a)
            cluster_atoms.pop(b)
            cluster_atoms[root(i)] = combined
    # Alignment is rebuilt as one deterministic unit; commit its trace once.
    # Per-pair fsync on the shared HDD would dominate this cache-only operation.
    trace_path = Path(out) / "alignment_decisions.jsonl"
    trace_tmp = trace_path.with_suffix(".jsonl.tmp")
    trace_tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in decision_records))
    trace_tmp.replace(trace_path)
    mapping, entities = {}, {}
    for r, ids in members.items():
        cid = "E_" + digest(sorted(local[i]["local_id"] for i in ids))[:20]
        names = list(dict.fromkeys(local[i]["name"] for i in ids))
        entities[cid] = {
            "entity_id": cid,
            "name": names[0],
            "aliases": names,
            "members": [local[i]["local_id"] for i in ids],
        }
        for i in ids:
            mapping[local[i]["local_id"]] = cid
    return mapping, entities, links, accepted_explicit, rejected, {"candidate_pairs": len(candidates), **filter_stats, **counts}
