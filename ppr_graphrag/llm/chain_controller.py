"""Grounded LLM initialization, entity linking and batched path checks."""

from pathlib import Path
import json

import numpy as np

from ppr_graphrag.core.text import norm
from ppr_graphrag.retrieval.chain_display import chain_text, reader_text
from ppr_graphrag.llm.chain_schemas import array, obj as schema_object

STRING = {"type": "string", "minLength": 1}


def answer_schema(paths, archives):
    refs = []
    for p in paths:
        if p["path"]:
            refs.append(schema_object({"kind": {"enum": ["path"]},
                "id": {"enum": [p["path_id"]]},
                "end_at": {"type": "integer", "minimum": 1, "maximum": len(p["path"])}}))
    if archives:
        refs.append(schema_object({"kind": {"enum": ["archive"]}, "id": {"enum": list(archives)}}))
    return schema_object(
        {
            "sufficient": {"type": "boolean"},
            "support_refs": array({"oneOf": refs}) if refs else {"type": "array", "maxItems": 0},
        }
    )


def check_schema(paths, archives):
    def prefixes(field):
        options = [schema_object({"path_id": {"enum": [p["path_id"]]},
                    "end_at": {"type": "integer", "minimum": 1, "maximum": len(p["path"])},
                    field: STRING}) for p in paths if p["path"]]
        return array({"oneOf": options}) if options else {"type": "array", "maxItems": 0}
    keyed_updates = {}
    for p in paths:
        positions = [0] + [i + 1 for i, step in enumerate(p["path"]) if step["to_node"] is not None]
        continuation = schema_object({
            "restart_at": {"enum": positions}, "question": STRING,
            "priority_relations": array(STRING, 12), "priority": {"enum": ["high", "medium", "low"]},
            "made_progress": {"type": "boolean"},
            "progress_fact_ids": array({"enum": list(dict.fromkeys(step["fact_id"] for step in p["path"]))}, len(p["path"]))
                if p["path"] else {"type": "array", "maxItems": 0},
            "reason": STRING,
        })
        continuing = array(continuation, 4)
        continuing["minItems"] = 1
        base = {"reason": STRING}
        keyed_updates[p["path_id"]] = {"oneOf": [
            schema_object(dict(base, action={"enum": ["continue"]}, continue_from=continuing)),
            schema_object(
                dict(base, action={"enum": ["archive", "discard"]}, continue_from={"type": "array", "maxItems": 0})
            ),
        ]}
    updates = schema_object(keyed_updates)
    return schema_object(
        {
            "updates": updates,
            "completed_parts": prefixes("question"),
            "retain_parts": prefixes("reason"),
            "answer_check": answer_schema(paths, archives),
        }
    )


def retained_check_schema(paths, archives):
    legacy = check_schema(paths, archives)
    updates = {}
    for p in paths:
        branches = legacy["properties"]["updates"]["properties"][p["path_id"]]["oneOf"]
        continuations = dict(branches[0]["properties"]["continue_from"])
        continuations["minItems"] = 0
        retained = array(schema_object({
            "end_at": {"type": "integer", "minimum": 1, "maximum": len(p["path"])},
            "status": {"enum": ["partial", "complete_local"]}, "question": STRING, "reason": STRING})) if p["path"] else {"type": "array", "maxItems": 0}
        updates[p["path_id"]] = schema_object({"retained_prefixes": retained,
            "continue_from": continuations, "reason": STRING})
    return schema_object({"updates": schema_object(updates), "answer_check": answer_schema(paths, archives)})


def lean_check_schema(paths, archives, mentions=None):
    selection=answer_schema(paths,archives)['properties']['support_refs']
    rows={}
    for p in paths:
        positions=[None,0]+[i+1 for i,step in enumerate(p['path']) if step['to_node'] is not None]
        rows[p['path_id']]=schema_object({'keep_to':{'enum':list(range(len(p['path'])+1))},
                                         'restart_at':{'enum':positions}})
    explore=dict(decision={'enum':['explore']},updates=schema_object(rows),
        relations=array(STRING,8),
        finished_mentions=array({'enum':mentions}) if mentions else dict(type='array',maxItems=0))
    return {'oneOf':[schema_object(dict(decision={'enum':['submit']},support_refs=selection)),schema_object(explore)]}


def decode_retained_check(obj, paths, archives):
    converted = {"updates": [], "completed_parts": [], "retain_parts": [], "answer_check": obj["answer_check"]}
    for pid, value in obj["updates"].items():
        retained = value["retained_prefixes"]
        for item in retained:
            if item["status"] == "complete_local":
                converted["completed_parts"].append({"path_id":pid,"end_at":item["end_at"],"question":item["question"],"reason":item["reason"]})
            elif item["status"] == "partial":
                converted["retain_parts"].append({"path_id":pid,"end_at":item["end_at"],"reason":item["reason"],"question":item["question"]})
            else:
                raise ValueError("Unknown retention status")
        converted["updates"].append(dict(path_id=pid, reason=value["reason"], continue_from=value["continue_from"],
            action="continue" if value["continue_from"] else "archive" if retained else "discard"))
    return validate_check(converted, paths, archives)


def prompt(name):
    return (Path(__file__).resolve().parents[2] / "prompts" / f"{name}.md").read_text()


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def compact_chain(chain, source_refs=False):
    """Complete directed chains; provenance is attached at the connection it supports."""
    facts = []
    for i, fact in enumerate(chain["evidence"]["facts"]):
        identity = []
        for link in fact["identity_before"]:
            identity.append({k: link[k] for k in
                ("names", "relation", "basis", "score", "qualifiers", "reason", "evidence", "textual_proof") if k in link})
        facts.append(dict(position=i+1, fact_id=fact["fact_id"],
            triple=fact["directed_triple"], source_triple=fact["source_triple"],
            qualifiers=fact["qualifiers"], identity_before=identity,
            kind="ATTRIBUTE" if fact["literal"] is not None else "ENTITY_RELATION",
            direction=fact["visit"]["direction"], doc_id=fact["doc_id"], quote=fact["quote"]))
    return dict(id=chain.get("path_id", chain.get("id")),
        current_question=chain.get("current_question", ""), local_parts=chain.get("local_parts", []),
        start_entity=facts[0]["triple"][0] if facts else chain.get("start_entity"),
        restart_positions=[0] + [i+1 for i,s in enumerate(chain["path"]) if s["to_node"] is not None],
        blocked=chain.get("blocked", False), terminal_attribute=bool(chain["path"] and chain["path"][-1]["to_node"] is None),
        exploration_origins=chain.get("parents", []), facts=facts)


def reader_chain(chain):
    view = compact_chain(chain, source_refs=True)
    for key in ("current_question", "local_parts", "restart_positions", "blocked", "terminal_attribute", "exploration_origins"):
        view.pop(key, None)
    return view


def validate_check(obj, paths, archives):
    lookup = {p["path_id"]: p for p in paths}
    if not isinstance(obj.get("updates"), list):
        raise ValueError("updates must be an array")
    seen = set()
    for update in obj["updates"]:
        pid = update.get("path_id")
        if pid not in lookup or pid in seen:
            raise ValueError("Unknown or duplicate path update")
        seen.add(pid)
        p = lookup[pid]
        action = update.get("action")
        if action not in ("continue", "archive", "discard") or not update.get("reason"):
            raise ValueError("Explicit path disposition/reason required")
        candidates = update.get("continue_from")
        if not isinstance(candidates, list) or bool(candidates) != (action == "continue"):
            raise ValueError("continue action and candidates disagree")
        for c in candidates:
            pos = c.get("restart_at")
            if not integer(pos) or not 0 <= pos <= len(p["path"]):
                raise ValueError("restart_at outside path")
            if pos and p["path"][pos - 1]["to_node"] is None:
                raise ValueError("Cannot restart at literal")
            if not isinstance(c.get("question"), str) or not c["question"].strip():
                raise ValueError("Missing remaining question")
            if c.get("priority") not in ("high", "medium", "low") or not isinstance(c.get("made_progress"), bool):
                raise ValueError("Bad priority/progress")
            if not isinstance(c.get("priority_relations"), list) or any(
                not isinstance(x, str) or not x.strip() for x in c["priority_relations"]
            ):
                raise ValueError("Bad relation hints")
            refs = c.get("progress_fact_ids", [])
            if not isinstance(refs, list) or not set(refs) <= {s["fact_id"] for s in p["path"][:pos]}:
                raise ValueError("Progress references outside retained prefix")
            if c["made_progress"] and (not refs or not c.get("reason")):
                raise ValueError("Progress needs retained evidence and reason")
    if seen != set(lookup):
        raise ValueError("Missing path dispositions")
    for key in ("completed_parts", "retain_parts"):
        if not isinstance(obj.get(key), list):
            raise ValueError(f"{key} must be array")
        for r in obj[key]:
            if r.get("path_id") not in lookup:
                raise ValueError("Unknown retained path")
            end = r.get("end_at")
            if not integer(end) or not 0 < end <= len(lookup[r["path_id"]]["path"]):
                raise ValueError("Invalid retained prefix position")
            if key == "completed_parts" and not r.get("question"):
                raise ValueError("Completed part requires local question")
    archived = {r["path_id"] for r in obj["completed_parts"] + obj["retain_parts"]}
    if any(u["action"] == "archive" and u["path_id"] not in archived for u in obj["updates"]):
        raise ValueError("Archive action without retained evidence prefix")
    validate_answer(obj.get("answer_check", {}), lookup, archives)
    return obj


def validate_answer(obj, paths, archives):
    if not isinstance(obj.get("sufficient"), bool) or not isinstance(obj.get("support_refs"), list):
        raise ValueError("Invalid sufficiency response")
    for ref in obj["support_refs"]:
        if ref.get("kind") == "archive":
            if ref.get("id") not in archives:
                raise ValueError("Unknown archive reference")
        elif ref.get("kind") == "path":
            if ref.get("id") not in paths:
                raise ValueError("Unknown support path")
            pos = ref.get("end_at")
            if not integer(pos) or not 0 < pos <= len(paths[ref["id"]]["path"]):
                raise ValueError("Invalid support prefix")
        else:
            raise ValueError("Unknown support kind")
    if obj["sufficient"] and not obj["support_refs"]:
        raise ValueError("Sufficient evidence cannot be empty")
    return obj


class Controller:
    def __init__(self, models, graph, local_vectors):
        self.models, self.graph = models, graph
        self.local = list(graph.data["local_entities"].values())
        self.local_vectors = local_vectors
        self.verification_feedback = []
        self.source_token_counts = {}

    def initialize(self, question):
        def validate(x):
            if not isinstance(x.get("start_mentions"), list) or not isinstance(x.get("priority_relations"), list):
                raise ValueError("Invalid initialization")
            if any(not isinstance(m, str) or not norm(m) or norm(m) not in norm(question) for m in x["start_mentions"]):
                raise ValueError("Start mentions must appear in original question")
            if any(not isinstance(r, str) or not r.strip() for r in x["priority_relations"]):
                raise ValueError("Invalid priority relations")
            expected = {"start_mentions", "priority_relations"}
            if self.models.initialization_variant == "one_shot":
                expected.add("expanded_relations")
                if not isinstance(x.get("expanded_relations"), list) or any(
                    not isinstance(r, str) or not r.strip() for r in x["expanded_relations"]
                ):
                    raise ValueError("Invalid expanded relations")
            if self.models.initialization_variant == "one_shot" and set(x) != expected:
                raise ValueError("Unexpected initialization fields")
            return x

        init = self.models.call("initialize", self.models.prompts["initialize"], "Question: " + question, validate, 1024)
        self.mention_names = init["start_mentions"]
        chosen, traces = [], []
        for mention in init["start_mentions"]:
            # Name vectors are aligned with local_entities. No question/answer
            # information enters this nearest-name retrieval or its scoring.
            vector = self.models.embed([norm(mention)])[0]
            scores = self.local_vectors @ vector
            exact = [i for i, e in enumerate(self.local) if norm(e["name"]) == norm(mention)]
            title_prefix = [i for i,e in enumerate(self.local) if len(mention)>=8 and
                any(e['name'].lower().startswith(mention.lower()+suffix) for suffix in (':',' ('))]
            preferred = sorted(set(exact+title_prefix),key=lambda i:-float(scores[i]))
            ranked = preferred + [int(i) for i in np.argsort(-scores,kind='stable') if int(i) not in set(preferred)]
            candidates, seen = [], set()
            for i in ranked:
                e = self.local[i]
                cid = self.graph.data["mapping"][e["local_id"]]
                if cid in seen:
                    continue
                seen.add(cid)
                candidates.append({"entity_id": cid, "name": e["name"], "doc_id": e["doc_id"],
                                   "local_id": e["local_id"], "score": float(scores[i]),
                                   "match": "normalized_name" if i in exact else "title_prefix" if i in title_prefix else "nv_embed_name"})
                if len(candidates) >= self.models.config["search"]["link_candidates"]:
                    break
            selected = [c["entity_id"] for c in candidates[:3]]
            chosen.extend(selected)
            traces.append({"mention": mention, "candidates": candidates,
                           "decision": {"selected": selected, "reason": "Deterministic name/alias retrieval; no LLM link call"},
                           "embedding_input": norm(mention)})
        hints, seen_hints = [], set()
        for hint in init["priority_relations"] + init.get("expanded_relations", []):
            key = norm(hint)
            if key not in seen_hints:
                seen_hints.add(key)
                hints.append(hint)
        if self.models.initialization_variant == "baseline":
            hints = init["priority_relations"]
        # Preserve raw fields and the actual merged guidance for diagnosis.
        return sorted(set(chosen)), hints, {"initialization": init, "links": traces,
                                           "merged_priority_relations": hints}

    def presentation(self, chain):
        view = compact_chain(chain, source_refs=True)
        view["prefix_sources"] = {
            str(i): self.graph.evidence(chain["path"][:i], chain["evidence"].get("anchor_local"))["doc_ids"]
            for i in range(1, len(chain["path"]) + 1)}
        return view

    def selection_context(self, chains):
        from .token_count import encoding
        doc_ids = sorted({d for c in chains for d in c["evidence"]["doc_ids"]})
        for d in doc_ids:
            if d not in self.source_token_counts:
                self.source_token_counts[d] = len(encoding().encode(json.dumps(self.graph.data["sources"][d], ensure_ascii=False), allowed_special="all"))
        return {"max_source_documents": self.models.config["reader"]["top_k"],
                "context_tokens": self.models.config["llm"]["context"],
                "reader_output_tokens": self.models.config["reader"]["output_tokens"],
                "document_tokens": {d:self.source_token_counts[d] for d in doc_ids},
                "note": "Count unique source IDs across selected prefixes, including identity evidence. Token sums exclude chain/prompt overhead; final program validates full input."}

    def validate_selection(self, answer, paths, archives, question):
        validate_answer(answer, paths, archives)
        selected = []
        for ref in answer["support_refs"]:
            if ref["kind"] == "archive":
                selected.append(archives[ref["id"]])
            else:
                full = paths[ref["id"]]
                path = full["path"][:ref["end_at"]]
                selected.append(dict(full, path=path, evidence=self.graph.evidence(path, full["evidence"].get("anchor_local"))))
        docs = sorted({d for p in selected for d in p["evidence"]["doc_ids"]})
        if len(docs) > self.models.config["reader"]["top_k"]:
            raise ValueError(f"Selected evidence needs {len(docs)} source articles, limit is 5. Keep useful complete chain units within budget; sufficient=false if no complete package fits.")
        data = reader_text(question, selected, [self.graph.data["sources"][d] for d in docs])
        messages = [{"role":"system","content":prompt("reader")},
                    {"role":"user","content":data}]
        if self.models.token_count(messages) + self.models.config["reader"]["output_tokens"] > self.models.config["llm"]["context"]:
            raise ValueError("Selected source package exceeds Reader token budget. Select fewer whole chains; do not assert sufficient unless that selected package answers all parts.")
        return answer

    def check(self, question, paths, archives):
        path_map = {f"p{i}": p["path_id"] for i, p in enumerate(paths)}
        archive_map = {f"a{i}": aid for i, aid in enumerate(archives)}
        wire_paths = [dict(p, path_id=k) for k, p in zip(path_map, paths)]
        wire_archives = {k:dict(archives[aid], id=k) for k, aid in archive_map.items()}
        self.wire_archive_ids = {v:k for k,v in archive_map.items()}
        checked = self._check_wire(question, wire_paths, wire_archives)
        for u in checked["updates"]:
            u["path_id"] = path_map[u["path_id"]]
        for key in ("completed_parts", "retain_parts"):
            for part in checked[key]:
                part["path_id"] = path_map[part["path_id"]]
        for ref in checked["answer_check"]["support_refs"] + checked.get("unsubmitted_support_refs", []):
            ref["id"] = (path_map if ref["kind"] == "path" else archive_map)[ref["id"]]
        return checked

    def _check_wire(self, question, paths, archives):
        text = ['Original question: ' + question, 'Named starting anchors: ' + '; '.join(self.mention_names), '', 'Previously saved evidence:']
        text += [chain_text(a, aid, self.graph) for aid,a in archives.items()] or ['None.']
        text += ['', 'New candidate chains:']
        text += [chain_text(p, p['path_id'], self.graph) for p in paths]
        if self.verification_feedback:
            text += ['', 'Reader feedback:', '\n'.join(f['reason'] for f in self.verification_feedback[-2:])]
        text += ['', 'First decide whether some prefixes jointly answer the original question.']
        allowed = {p['path_id']:p for p in paths}
        def validate(x):
            if x.get('decision') not in ('submit','explore'):
                raise ValueError('decision must be submit or explore')
            answer = dict(sufficient=x['decision']=='submit',support_refs=x.get('support_refs',[]))
            validate_answer(answer, allowed, archives)
            updates, completed, retained = [], [], []
            if answer['sufficient']:
                # Unselected paths remain available if Reader rejects the package.
                selections={r['id']:r['end_at'] for r in answer['support_refs'] if r['kind']=='path'}
                raw=[dict(id=p['path_id'],keep_to=selections.get(p['path_id'],0),status='partial',
                          contribution=question,restart_at=None,remaining=question,relations=[]) for p in paths]
            else:
                keyed=x.get('updates',{})
                if not isinstance(keyed,dict) or set(keyed)!=set(allowed):
                    raise ValueError('updates must contain exactly the supplied new chain labels')
                rows=[keyed[p['path_id']] for p in paths]
                hints=x.get('relations',[])
                if not isinstance(hints,list) or any(not isinstance(h,str) or not h for h in hints):
                    raise ValueError('relations must be relation hints, not answers')
                raw=[]
                for p,row in zip(paths,rows):
                    end=row.get('keep_to',0)
                    contribution=(' → '.join(str(v) for v in p['evidence']['facts'][end-1]['directed_triple'])
                                  if integer(end) and 0<end<=len(p['path']) else 'No useful prefix')
                    raw.append(dict(row,id=p['path_id'],status='partial',contribution=contribution,
                                    remaining=question,relations=hints))
            for u in raw:
                p=allowed[u['id']]; end=u.get('keep_to',0); pos=u.get('restart_at')
                if not integer(end) or not 0<=end<=len(p['path']):
                    raise ValueError('keep_to outside chain')
                reason=u.get('contribution') or 'No useful prefix'
                if end:
                    part=dict(path_id=u['id'],end_at=end,question=reason,reason=reason)
                    (completed if u.get('status')=='complete_local' else retained).append(part)
                cs=[]
                if integer(pos) and 0<pos<=len(p['path']) and p['path'][pos-1]['to_node'] is None:
                    pos=None
                if pos is not None:
                    if not integer(pos) or not 0<=pos<=len(p['path']) or (pos and p['path'][pos-1]['to_node'] is None):
                        raise ValueError('restart_at must be an entity position, not a terminal detail')
                    cs=[dict(restart_at=pos,question=u.get('remaining') or question,
                        priority_relations=u.get('relations',[]),priority='medium',made_progress=bool(pos),
                        progress_fact_ids=[step['fact_id'] for step in p['path'][:pos]],reason=reason)]
                updates.append(dict(path_id=u['id'],action='continue' if cs else 'archive' if end else 'discard',
                    reason=reason,continue_from=cs))
            finished=x.get('finished_mentions',[])
            if not isinstance(finished,list) or any(m not in self.mention_names for m in finished):
                raise ValueError('finished_mentions must use named starting anchors')
            result=dict(updates=updates,completed_parts=completed,retain_parts=retained,answer_check=answer,
                        finished_mentions=finished)
            validate_check(result,paths,archives)
            try:
                self.validate_selection(answer,allowed,archives,question)
            except ValueError as exc:
                result.update(selection_rejected=str(exc),unsubmitted_support_refs=answer['support_refs'],
                              answer_check=dict(sufficient=False,support_refs=[]))
            return result
        return self.models.call('check',prompt('check'),'\n'.join(text),validate,2048, schema=lean_check_schema(paths,archives, self.mention_names))

    def assemble(self, question, archives):
        # Short request-local IDs avoid huge schema literals; disk IDs remain stable.
        mapping = {f"a{i}": aid for i, aid in enumerate(archives)}
        wire = {short: dict(archives[aid], id=short) for short, aid in mapping.items()}
        inverse = {v:k for k,v in mapping.items()}
        text = ['Original question: '+question, '', 'Saved evidence:']
        text += [chain_text(a,k,self.graph) for k,a in wire.items()]
        if self.verification_feedback:
            text += ['Reader feedback: ' + self.verification_feedback[-1]['reason']]
        result = self.models.call('assemble',prompt('assemble'),'\n'.join(text),
            lambda x:self.validate_selection(dict(sufficient=x.get('sufficient'),support_refs=x.get('support_refs',[])),{},wire,question),1024, schema=answer_schema([],wire))
        for ref in result["support_refs"]:
            ref["id"] = mapping[ref["id"]]
        return result

    def partial_selection(self, question, archives):
        """Deterministic fallback from already retained chains; never infer completeness."""
        selected, covered, skipped = [], set(), []
        ordered = sorted(archives.values(), key=lambda a: (
            -int(a["completed"]), len(a["evidence"]["doc_ids"]), -len(a["path"]), a["id"]))
        for chain in ordered:
            contributions = {p["question"] for p in chain["local_parts"]}
            if selected and contributions and contributions <= covered:
                skipped.append(dict(id=chain['id'],reason='Contribution already covered'))
                continue
            candidate = {"sufficient": False, "support_refs": selected + [{"kind":"archive", "id":chain["id"]}]}
            try:
                self.validate_selection(candidate, {}, archives, question)
            except ValueError as exc:
                skipped.append(dict(id=chain['id'],reason=str(exc),doc_ids=chain['evidence']['doc_ids']))
                continue
            selected = candidate["support_refs"]
            covered |= contributions
        from ppr_graphrag.llm.structured_runtime import append
        append(self.models.out/'partial_selection.jsonl',dict(selected=selected,skipped=skipped))
        return {"sufficient": False, "support_refs": selected}
