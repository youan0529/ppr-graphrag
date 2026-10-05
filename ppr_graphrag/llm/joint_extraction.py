"""Five-way fact routing with per-row audit and first-attempt partial recovery."""
import copy
import json
import re
from functools import lru_cache
from pathlib import Path

from ppr_graphrag.llm.structured_runtime import digest, dump, file_digest

ROOT = Path(__file__).resolve().parents[2]
TYPE_SOURCE = (ROOT / 'prompts/types.md').read_text()
TYPE_PROMPT = '\n'.join(line for line in TYPE_SOURCE.splitlines() if re.match(r'^[A-Z][A-Za-z /]+:', line))
TYPES = [line.split(':', 1)[0] for line in TYPE_PROMPT.splitlines()]
KINDS = ['ENTITY_LINK', 'ENTITY_TO_DETAIL', 'DETAIL_TO_ENTITY', 'TEXT_FACT', 'SAME_ENTITY']
QUALIFIERS = ['time', 'start_time', 'end_time', 'duration', 'location', 'condition', 'attribution']
PURE_VALUES = {'Date or time', 'Duration', 'Number or quantity', 'Truth value', 'Identifier or address'}


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


STRING = {'type': 'string'}
ENDPOINT = {'type': 'array', 'items': [STRING, {'enum': TYPES}, STRING], 'minItems': 3, 'maxItems': 3}
COMMON = {
    'qualifiers': {'type': 'object', 'properties': {k: STRING for k in QUALIFIERS}, 'additionalProperties': False},
}
TRIPLE = dict(subject=ENDPOINT, relation=STRING, object=ENDPOINT, **COMMON)
SCHEMA = obj({'facts': {'type': 'array', 'maxItems': 96, 'items': {'oneOf': [
    obj(dict(kind={'enum': ['ENTITY_LINK', 'ENTITY_TO_DETAIL', 'SAME_ENTITY']}, **TRIPLE)),
    obj(dict(kind={'const': 'DETAIL_TO_ENTITY'}, **TRIPLE,
             reverse_relation={'type': ['string', 'null']}, text=STRING)),
    obj(dict(kind={'const': 'TEXT_FACT'}, text=STRING, **COMMON)),
]}}})
INVERSE_SCHEMA = obj({'inverses': {'type': 'array', 'items': obj({
    'id': {'type': 'integer', 'minimum': 0}, 'reverse_relation': {'type': ['string', 'null']}
})}})


def validate_fact(raw):
    if not isinstance(raw, dict):
        raise ValueError('Fact must be an object')
    fact = copy.deepcopy(raw)
    kind = fact.get('kind')
    if kind not in KINDS:
        raise ValueError('Unknown routing kind')
    expected = {'kind', 'qualifiers'}
    expected |= {'text'} if kind == 'TEXT_FACT' else {'subject', 'relation', 'object'}
    if kind == 'DETAIL_TO_ENTITY':
        expected |= {'reverse_relation', 'text'}
    if set(fact) != expected:
        raise ValueError('Wrong fields for ' + kind)
    if kind != 'TEXT_FACT':
        for role in ['subject', 'object']:
            ep = fact[role]
            if (not isinstance(ep, list) or len(ep) != 3 or not all(isinstance(x, str) for x in ep)
                    or not ep[0].strip() or ep[1] not in TYPES):
                raise ValueError('Endpoint must be [nonempty text, semantic type, disambiguator]')
            fact[role] = [x.strip() for x in ep]
        if not isinstance(fact['relation'], str) or not fact['relation'].strip():
            raise ValueError('Empty relation')
        fact['relation'] = fact['relation'].strip()
        entity_roles = (['subject', 'object'] if kind in ['ENTITY_LINK', 'SAME_ENTITY']
                        else ['subject'] if kind == 'ENTITY_TO_DETAIL' else ['object'])
        for role in entity_roles:
            if fact[role][1] in PURE_VALUES:
                raise ValueError(f'{kind}: {role} is a terminal value in an entity role; correct the assertion/route, not just its label')
        if kind == 'SAME_ENTITY' and fact['subject'][1] != fact['object'][1]:
            raise ValueError('Aliases must share the type of their referent; identity is not an identifier value')
    if 'text' in fact and (not isinstance(fact['text'], str) or not fact['text'].strip()):
        raise ValueError('Standalone fact text required')
    if kind == 'DETAIL_TO_ENTITY':
        reverse = fact['reverse_relation']
        if reverse is not None and (not isinstance(reverse, str) or not reverse.strip()):
            raise ValueError('Reverse predicate must be nonempty or null')
    q = fact['qualifiers']
    if not isinstance(q, dict) or not set(q) <= set(QUALIFIERS) or any(not isinstance(v, str) or not v.strip() for v in q.values()):
        raise ValueError('Invalid qualifier')
    return fact


def audit_facts(value):
    if not isinstance(value, dict) or set(value) != {'facts'} or not isinstance(value['facts'], list) or len(value['facts']) > 96:
        raise ValueError('Expected facts array (at most 96)')
    clean, rejected, seen = [], [], set()
    for index, raw in enumerate(value['facts']):
        try:
            fact = validate_fact(raw)
            key = digest(fact)
            if key not in seen:
                seen.add(key)
                clean.append(fact)
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append(dict(index=index, error=str(exc), fact=raw))
    return dict(facts=clean, rejected=rejected, input_rows=len(value['facts']))


def validate_facts(value):
    audit = audit_facts(value)
    if audit['rejected']:
        raise ValueError(str(audit['rejected'][0]))
    return {'facts': audit['facts']}


def audit_inverses(value, expected):
    """Validate each correspondence/representation; this is not semantic review."""
    entries = value.get('inverses') if isinstance(value, dict) else None
    if not isinstance(entries, list):
        return {'accepted': [], 'rejected': [{'error': 'Expected inverses array'}]}
    counts = {}
    for row in entries:
        if isinstance(row, dict) and type(row.get('id')) is int:
            counts[row['id']] = counts.get(row['id'], 0) + 1
    accepted, rejected = [], []
    for row in entries:
        try:
            if not isinstance(row, dict) or set(row) != {'id', 'reverse_relation'}:
                raise ValueError('Expected id and reverse_relation')
            index = row['id']
            if type(index) is not int or index not in expected or counts[index] != 1:
                raise ValueError('Unknown or duplicate relation id')
            reverse = row['reverse_relation']
            if isinstance(reverse, str):
                reverse = reverse.strip()
                if reverse.lower() in {'null', 'none'}:
                    reverse = None
                elif not reverse:
                    raise ValueError('Empty predicate; use null when unavailable')
                elif re.search(r'\b[XY]\b|\{(?:from|to|subject|object)\}', reverse):
                    raise ValueError('Use a short edge label such as is location of, without endpoint placeholders')
            elif reverse is not None:
                raise ValueError('Predicate must be a string or null')
            accepted.append({'id': index, 'reverse_relation': reverse})
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({'row': row, 'error': str(exc)})
    return {'accepted': accepted, 'rejected': rejected}


def extract_document(doc, models, folder):
    folder = Path(folder)
    saved = folder / 'extraction.json'
    content_sha = digest({k: doc[k] for k in ['doc_id', 'title', 'text']})
    if saved.exists():
        row = json.loads(saved.read_text())
        if row['content_sha'] != content_sha:
            raise ValueError('Source changed after extraction')
        return row
    payload = 'Title: ' + doc['title'] + '\n\n' + doc['text']
    prompt = (ROOT / 'prompts/joint.md').read_text()
    models.prompts['joint'] = prompt
    calls_path = folder / 'calls.jsonl'
    start_offset = calls_path.stat().st_size if calls_path.exists() else 0
    error = None
    try:
        value = models.call('joint', prompt, payload, lambda v: validate_facts(v),
                            models.config['extraction']['joint_output_tokens'], schema=SCHEMA)
    except ValueError as exc:
        if 'invalid model output after bounded retries' not in str(exc):
            raise
        error = str(exc)
        value = None
    audits = []
    if calls_path.exists():
        with calls_path.open('rb') as handle:
            handle.seek(start_offset)
            events = [json.loads(line) for line in handle if line.strip()]
        for event in events:
            if event['stage'] != 'joint':
                continue
            response = event.get('response') or {}
            try:
                if not response.get('done') or response.get('done_reason') == 'length':
                    raise ValueError('Incomplete response; no partial JSON recovery')
                audit = audit_facts(json.loads(response['message']['content']))
            except (ValueError, KeyError, TypeError) as exc:
                audit = dict(facts=[], rejected=[], parse_error=str(exc))
            audits.append(dict(attempt=event['attempt'], **audit))
    dump(folder / 'validation.json', dict(attempts=audits, final_error=error))
    status = 'complete'
    if value is None:
        first = next((a for a in audits if a['attempt'] == 0), {})
        if not first.get('facts'):
            raise ValueError(error + '; no individually valid rows in first response')
        value = {'facts': first['facts']}
        status = 'partial'
    row = dict(doc_id=doc['doc_id'], content_sha=content_sha, **value,
               extraction_status=status, recovery_policy='first_attempt_valid_rows_only' if status == 'partial' else None,
               validation_note='Structural/routing checks do not establish semantic correctness.')
    dump(saved, row)
    return row


def role_inverse(relation):
    """Reverse explicit role predicates without guessing either participant's gender or order."""
    # Familiar broad inverses for bare kinship predicates; never infer the other endpoint's gender.
    simple = re.fullmatch(r'(?:(is|was) )?(?:a |an |the )?(mother|father|parent|son|daughter|child) of', relation.strip().lower())
    if simple:
        opposite = 'child' if simple[2] in ('mother', 'father', 'parent') else 'parent'
        return ('was' if simple[1] == 'was' else 'is') + ' ' + opposite + ' of'
    roles = r'(?:mother|father|parent|son|daughter|child|wife|husband|spouse|brother|sister|sibling|uncle|aunt|nephew|niece|grandfather|grandmother|grandparent|grandson|granddaughter|grandchild|great-grandfather|great-grandmother|great-grandparent|great-grandson|great-granddaughter|great-grandchild)'
    modifiers = r'(?:(?:first|second|third|fourth|fifth|sixth|seventh|eighth|eldest|oldest|elder|older|youngest|younger|full|half|only|maternal|paternal|biological|adoptive) )*'
    pattern = r'^(?:(is|was) )?(?:a |an |the )?(' + modifiers + roles + r') (?:of|to)$'
    match = re.fullmatch(pattern, relation.strip().lower())
    if match:
        return ('had' if match[1] == 'was' else 'has') + ' as ' + match[2]
    match = re.fullmatch(r'^(has|had) (?:a |an |the )?(' + modifiers + roles + r')$', relation.strip().lower())
    if match:
        return ('was' if match[1] == 'had' else 'is') + ' ' + match[2] + ' of'
    return None


@lru_cache(maxsize=1)
def reused_inverse_rows(path, expected_sha=None):
    if path and (not Path(path).exists() or (expected_sha and file_digest(path) != expected_sha)):
        raise ValueError('Inverse reuse source missing or changed')
    data = json.loads(Path(path).read_text()) if path else {'rows': []}
    return {digest({'relation': row['forward'], 'examples': row.get('examples', [])}): row for row in data['rows']}


def inverse_batch(records, models):
    """One low call, with bounded retries for unresolved rows only.

    Accepted rows are checkpointed independently, including explicit nulls.
    The original four-row IDs remain stable when retrying a subset.
    """
    folder = Path(models.out)
    identity = digest({'records': records, 'prompt': (ROOT / 'prompts/inverse.md').read_text(),
                       'model': models.config['llm']['digest'], 'think': 'low',
                       'reuse_sha': models.config['extraction'].get('inverse_reuse_sha256')})
    checkpoint = folder / 'inverse_rows.json'
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    if saved and saved['identity'] != identity:
        raise ValueError('Inverse checkpoint identity changed')
    results = {int(k): v for k, v in (saved or {}).get('accepted', {}).items()}
    audits = (saved or {}).get('audits', [])
    def save():
        dump(checkpoint, {'identity': identity, 'accepted': results, 'audits': audits})
    reuse = reused_inverse_rows(models.config['extraction'].get('inverse_reuse_path', ''),
                               models.config['extraction'].get('inverse_reuse_sha256'))
    for i, record in enumerate(records):
        cached = reuse.get(digest(record))
        if i not in results and cached and cached.get('inverse'):
            results[i] = copy.deepcopy(cached)
            results[i]['reused_existing_inverse'] = True
        if i not in results and role_inverse(record['relation']):
            results[i] = dict(forward=record['relation'], inverse=role_inverse(record['relation']),
                              reason='deterministic role inverse', method='role_rule', examples=record['examples'])
    save()
    original_attempts = models.config['llm']['attempts']
    # Outer row recovery controls the budget. Runtime may not retry the full batch.
    models.config['llm']['attempts'] = 1
    try:
        for attempt in range(models.config['extraction'].get('inverse_row_attempts', 2)):
            pending = {i: r for i, r in enumerate(records) if i not in results}
            if not pending:
                break
            blocks = []
            for i, record in pending.items():
                lines = []
                for example_index, fact in enumerate(record['examples']):
                    subject, obj = fact['subject'], fact['object']
                    prefix = f"Relation {i}: " if example_index == 0 else "  "
                    lines.append(prefix + f"{subject[0]} [{subject[1]}] → {record['relation']} → {obj[0]} [{obj[1]}]")
                blocks.append('\n'.join(lines))
            payload = '\n\n'.join(blocks)
            if attempt:
                payload += '\n\nReturn only these remaining relation IDs. Previously accepted rows are already saved. Use a short ordinary label; no X/Y placeholders.'
                payload += '\nPrevious format problems: ' + json.dumps(audits[-1].get('rejected', []), ensure_ascii=False)[:1000]
            try:
                audit = models.call('inverse', (ROOT / 'prompts/inverse.md').read_text(), payload,
                                    lambda value: audit_inverses(value, pending),
                                    models.config['extraction']['inverse_output_tokens'], schema=INVERSE_SCHEMA)
            except ValueError as exc:
                audit = {'accepted': [], 'rejected': [{'error': str(exc)}]}
            audits.append({'attempt': attempt, 'requested_ids': list(pending), **audit})
            for row in audit['accepted']:
                record = pending[row['id']]
                results[row['id']] = dict(forward=record['relation'], inverse=row['reverse_relation'],
                    reason='Reverse label generated from supplied endpoints; no separate example or review',
                    method='llm_low_single_pass', examples=record['examples'],
                    generation_status='available' if row['reverse_relation'] is not None else 'unavailable')
            save()
    finally:
        models.config['llm']['attempts'] = original_attempts
    unresolved = [i for i in range(len(records)) if i not in results]
    dump(folder / 'inverse_validation.json', {'accepted': len(results), 'unresolved_ids': unresolved,
         'audits': audits, 'note': 'Format/correspondence only, not semantic certification.'})
    # Failures remain explicit per relation. Other successful rows still enter the graph.
    return [results[i] if i in results else dict(forward=r['relation'], inverse=None,
            reason='No valid row after bounded format recovery', method='unresolved_format',
            examples=r['examples'], generation_status='format_failed') for i, r in enumerate(records)]


def guard_inverse_materialization(row):
    # The row validator handles representational failures. Do not reject useful
    # approximate labels through a symmetry whitelist or endpoint-name heuristics.
    return copy.deepcopy(row)
