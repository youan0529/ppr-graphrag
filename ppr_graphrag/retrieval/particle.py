"""Persistent two-step frontiers, particle effort, cached dispositions and joint checks."""
import copy
import json
import math
from pathlib import Path
import random
from ppr_graphrag.llm.structured_runtime import digest,dump,append
from ppr_graphrag.retrieval.evidence import tuple_tree,compact_archives
from ppr_graphrag.core.text import norm
import re


def retrieve(query_id,question,graph,controller,models,cfg,out):
    folder=Path(out)/'queries'/query_id;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'checkpoint.json';sc=cfg['search'];B=sc['effort_batch']
    rng=random.Random(int(digest([cfg['seed'],query_id])[:16],16))
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved.get('result'):return saved['result']
        fronts=saved['frontiers'];chains=saved['chains'];archives=saved['archives']
        initialization=saved['initialization'];finished=set(saved['finished_mentions'])
        controller.verification_feedback=saved['feedback'];rng.setstate(tuple_tree(saved['rng']))
        first=saved['round']+1;exploration=saved['exploration_steps'];tried=set(saved['submitted'])
    else:
        _,hints,initialization=controller.initialize(question)
        fronts={};chains={};archives={};finished=set();first=1;exploration=0;tried=set()
        for link in initialization['links']:
            candidates=link['candidates'];maximum=max((c['score'] for c in candidates),default=1)
            for rank,c in enumerate(candidates):
                key=digest([link['mention'],c['entity_id'],[]])[:24]
                prior=math.exp((c['score']-maximum)/sc['anchor_temperature'])
                if c['match'] in ('normalized_name','title_prefix'):prior=max(prior,1.)
                fronts[key]=dict(id=key,mention=link['mention'],node=c['entity_id'],prefix=[],
                    anchor=c['local_id'],hints=hints,epsilon=0.,base=prior,rank=rank,
                    occurrences={},processed=0,total=0,blocked=0,born=0,epoch='')
        dump(folder/'initialization.json',initialization)
    controller.mention_names=initialization['initialization']['start_mentions']
    title_docs={norm(d['title']):did for did,d in graph.data['sources'].items()}

    def source_probabilities(node,path,priority,front):
        edges,base=graph.probabilities(node,path,priority,front['epsilon'])
        local=path[-1]['to_local'] if path else front['anchor']
        name=graph.data['local_entities'][local]['name'] if local else ''
        primary=title_docs.get(norm(name))
        if not path:
            anchor_doc=graph.data['local_entities'][front['anchor']]['doc_id']
            title=graph.data['sources'][anchor_doc]['title'];mention=front['mention']
            if norm(title)==norm(mention) or ('(' not in mention and norm(re.sub(r'\([^)]*\)','',title))==norm(mention)):
                primary=anchor_doc
        used={step['fact_id'] for step in path}
        indexes=[i for i,e in enumerate(edges) if graph.facts[e['fact_id']]['doc_id']==primary and e['fact_id'] not in used]
        if not indexes:return edges,base,None
        specific=[0.]*len(edges);total=sum(base[i] for i in indexes)
        for i in indexes:specific[i]=base[i]/total if total else 1/len(indexes)
        share=sc['source_share'] if any(base) else 1.
        return edges,[(1-share)*p+share*q for p,q in zip(base,specific)],primary


    def evidence(path,anchor):return graph.evidence(path,anchor)
    def archive(chain,end,contribution,complete=False):
        path=chain['path'][:end];aid='a'+digest(path)[:24]
        if aid not in archives:
            archives[aid]=dict(id=aid,path=path,evidence=evidence(path,chain['anchor']),
                completed=False,active=True,local_parts=[],local_questions=[],retention_reasons=[],mentions=[])
        a=archives[aid];a['completed']|=complete
        part=dict(question=contribution,status='complete_local' if complete else 'partial')
        if part not in a['local_parts']:a['local_parts'].append(part)
        if contribution not in a['local_questions']:a['local_questions'].append(contribution)
        a['mentions']=list(dict.fromkeys(a['mentions']+[fronts[s]['mention'] for s in chain['owners']]))
        return aid
    def candidate(answer,rnd):
        selected=[archives[r['id']] for r in answer['support_refs']]
        return dict(query_id=query_id,original_question=question,selected_paths=selected,
            supporting_doc_ids=list(dict.fromkeys(d for a in selected for d in a['evidence']['doc_ids'])),
            fact_ids=list(dict.fromkeys(s['fact_id'] for a in selected for s in a['path'])),
            evidence_complete=answer['sufficient'],round_count=rnd,exploration_steps=exploration,
            initialization=initialization,stop_reason='checker_sufficient' if answer['sufficient'] else 'budget_exhausted')
    def active_archives():
        return {k:a for k,a in archives.items() if a.get('active',True)}
    result=None
    for rnd in range(first,sc['rounds']+1):
        eligible=[s for s in fronts.values() if s['mention'] not in finished and
            len(s['prefix'])<sc['steps']*sc['rounds'] and (rnd>1 or s['rank']<3)]
        priority,rel_trace=graph.classify([h for s in eligible for h in s['hints']],models,sc['relation_threshold'])
        counts={};samples=[];probabilities={}
        retry_blocked=[s for s in eligible if s['blocked']>=B and not s['total']]
        for s in eligible:
            epoch=digest([sorted(priority),s['epsilon']])[:16]
            if s['epoch']!=epoch:
                s['epoch']=epoch;s['occurrences']={};s['processed']=0;s['total']=0;s['blocked']=0
            s['processed']=sum(n for key,n in s['occurrences'].items() if chains[key].get('checked'))
        # Give newly discovered useful prefixes one small batch before broad exploration.
        fresh=[s for s in eligible if s['prefix'] and s['total']==0]
        rng.shuffle(fresh)
        retry_blocked.sort(key=lambda s:(s['rank'],-s['base']))
        scheduled=(fresh+retry_blocked)[:max(1,sc['particles']//(2*B))]
        budget=sc['particles']
        while eligible and budget>0:
            if scheduled:s=scheduled.pop(0)
            else:
                warm=[s for s in eligible if s['prefix'] or s['rank']<3 or s.get('attempted')]
                cold=[s for s in eligible if s not in warm]
                weights=[s['base']*B/(B+s['processed']) for s in warm]
                # Untried candidates form ONE option: their count must not swamp the good anchors.
                choices=list(warm)
                if cold:
                    choices.append(None);weights.append(sum(s['base'] for s in cold)/len(cold))
                s=rng.choices(choices,weights=weights,k=1)[0]
                if s is None:s=rng.choices(cold,weights=[s['base'] for s in cold],k=1)[0]
            s['attempted']=True
            amount=min(B,budget);budget-=amount
            for _ in range(amount):
                path=list(s['prefix']);node=s['node'];new=0;blocked=False
                for stepno in range(sc['steps']):
                    edges,probs,primary=source_probabilities(node,path,priority,s)
                    pk=digest([node,[t['fact_id'] for t in path],s['epoch'],primary])[:20]
                    probabilities[pk]=dict(node=node,used=[t['fact_id'] for t in path],epsilon=s['epsilon'],
                                           edges=edges,probabilities=probs,preferred_source=primary,source_share=sc['source_share'])
                    if not any(probs):blocked=True;break
                    step=dict(rng.choices(edges,weights=probs,k=1)[0]);path.append(step);new+=1
                    if step['to_node'] is None:break
                    node=step['to_node']
                exploration+=new;counts[s['id']]=counts.get(s['id'],0)+1
                if new==0:
                    s['blocked']+=1
                    samples.append(dict(frontier=s['id'],new_steps=0,blocked=True))
                    continue
                key='p'+digest(path)[:24]
                if key not in chains:
                    chains[key]=dict(path_id=key,path=path,anchor=s['anchor'],
                        evidence=evidence(path,s['anchor']),start_node=path[0]['from_node'],
                        start_entity=graph.data['entities'][path[0]['from_node']]['name'],
                        current_question=question,owners=[],parents=[],blocked=blocked,
                        terminal_attribute=path[-1]['to_node'] is None,checked=False,discovered=rnd)
                chain=chains[key]
                if s['id'] not in chain['owners']:chain['owners'].append(s['id'])
                s['total']+=1;s['occurrences'][key]=s['occurrences'].get(key,0)+1
                if chain['checked']:s['processed']+=1
                samples.append(dict(frontier=s['id'],path_id=key,new_steps=new,already_checked=chain['checked']))
            # A blocked policy must relax before being interpreted as semantic failure.
            if s['blocked']>=B and not s['total']:
                eligible=[x for x in eligible if x['id']!=s['id']]
        view=active_archives()
        # Keep the live check bounded; all retained records remain on disk.
        display_ids=sorted(view,key=lambda a:(-int(view[a]['completed']),len(view[a]['evidence']['doc_ids']),-len(view[a]['path'])))[:sc['archive_display']]
        displayed={a:view[a] for a in display_ids}
        pending=[c for c in chains.values() if not c['checked'] and
                 any(fronts[s]['mention'] not in finished for s in c['owners'])]
        # Prefer continuations, balance named starting tasks, preserve pending leftovers.
        pending.sort(key=lambda c:(-max(len(fronts[s]['prefix']) for s in c['owners']),c['discovered'],c['path_id']))
        pools={m:[] for m in controller.mention_names if m not in finished}
        for c in pending:
            for m in dict.fromkeys(fronts[s]['mention'] for s in c['owners']):
                if m in pools:pools[m].append(c)
        selected=[];selected_ids=set();capacity=max(1,sc['check_batch']-len(displayed))
        while len(selected)<capacity and any(pools.values()):
            for group in pools.values():
                while group and group[0]['path_id'] in selected_ids:group.pop(0)
                if group and len(selected)<capacity:
                    c=group.pop(0);selected.append(c);selected_ids.add(c['path_id'])
        checked=None;failures=[];answer=dict(sufficient=False,support_refs=[])
        if selected:
            try:
                checked=controller.check(question,selected,displayed)
                lookup={c['path_id']:c for c in selected}
                for c in selected:c['checked']=True
                for part in checked['completed_parts']+checked['retain_parts']:
                    archive(lookup[part['path_id']],part['end_at'],part.get('question',question),part in checked['completed_parts'])
                for ref in checked['answer_check']['support_refs']+checked.get('unsubmitted_support_refs',[]):
                    if ref['kind']=='path':archive(lookup[ref['id']],ref['end_at'],question)
                answer=copy.deepcopy(checked['answer_check'])
                for ref in answer['support_refs']:
                    if ref['kind']=='path':ref.update(kind='archive',id=archive(lookup[ref['id']],ref.pop('end_at'),question))
                for update in checked['updates']:
                    c=lookup[update['path_id']]
                    for continuation in update['continue_from']:
                        prefix=c['path'][:continuation['restart_at']]
                        node=prefix[-1]['to_node'] if prefix else c['start_node']
                        for owner in c['owners']:
                            parent=fronts[owner];fid=digest([parent['mention'],node,prefix])[:24]
                            hints=continuation['priority_relations'] or parent['hints']
                            if fid not in fronts:
                                fronts[fid]=dict(id=fid,mention=parent['mention'],node=node,prefix=prefix,
                                    anchor=c['anchor'],hints=hints,epsilon=0. if prefix else parent['epsilon'],
                                    base=parent['base'],rank=0,occurrences={},processed=0,total=0,blocked=0,born=rnd,epoch='')
                            else:fronts[fid]['hints']=list(dict.fromkeys(fronts[fid]['hints']+hints))
                # Mission completion is explicit, and must have a retained contribution.
                saved_mentions={m for a in archives.values() for m in a['mentions']}
                finished.update(m for m in checked.get('finished_mentions',[]) if m in saved_mentions)
            except ValueError as exc:
                failures.append(dict(stage='check',error=str(exc)))
        elif displayed:
            try:answer=controller.assemble(question,displayed)
            except ValueError as exc:failures.append(dict(stage='assemble',error=str(exc)))
        compact_archives(archives)
        view=active_archives()
        if not answer['support_refs']:answer=controller.partial_selection(question,view)
        result=candidate(answer,rnd)
        if view or rnd==sc['rounds']:  # Let source-grounded Reader judge every newly retained package.
            from ppr_graphrag.pipelines.answer_particle import read_answer
            package=digest([result['supporting_doc_ids'],[c['id'] for c in result['selected_paths']]])
            if package not in tried and result['supporting_doc_ids']:
                tried.add(package)
                result=read_answer(result,graph.data,models,cfg)
                dump(folder/f'round_{rnd:02d}_verification.json',result)
                if not result['evidence_complete']:
                    reason=result.get('reader_output',{}).get('reason',result['stop_reason'])
                    controller.verification_feedback.append(dict(round=rnd,reason=reason,
                        rejected_archive_ids=[c['id'] for c in result['selected_paths']],source_doc_ids=result['supporting_doc_ids']))
                    reopened=result.get('reader_output',{}).get('reopen_chain_ids',[])
                    for aid in reopened:
                        archives[aid]['active']=False
                        finished.difference_update(archives[aid]['mentions'])
                    if not reopened:
                        finished.difference_update(m for c in result['selected_paths'] for m in c['mentions'])
                    # Previously discarded chains can be relevant to a newly identified gap.
                    for c in chains.values():
                        if not any(a['path']==c['path'] and a.get('active',True) for a in archives.values()):c['checked']=False
            elif package in tried:
                result['evidence_complete']=False
        # Exploration effort is counted only for processed, actually walked outcomes.
        for s in fronts.values():
            s['processed']=sum(n for k,n in s['occurrences'].items() if chains[k]['checked'])
            if counts.get(s['id']) and (s['blocked']>=B or s['processed']>=B):
                s['epsilon']=round(min(sc['epsilon_max'],s['epsilon']+sc['delta']),10)
        result.update(verification_feedback=controller.verification_feedback,
            sampled_doc_ids=sorted({d for c in chains.values() for d in c['evidence']['doc_ids']}),
            archived_doc_ids=sorted({d for a in archives.values() for d in a['evidence']['doc_ids']}),
            deferred_check_failures=failures)
        done=result.get('reader_executed') and result.get('evidence_complete') or rnd==sc['rounds']
        dump(folder/f'round_{rnd:02d}.json',dict(round=rnd,counts=counts,samples=samples,
            probabilities=probabilities,relations=rel_trace,check=checked,failures=failures,
            checked_path_ids=[c['path_id'] for c in selected],displayed_archives=display_ids,
            pending=len([c for c in chains.values() if not c['checked']]),finished_mentions=sorted(finished),
            frontiers=fronts,answer=answer))
        dump(checkpoint,dict(round=rnd,frontiers=fronts,chains=chains,archives=archives,
            initialization=initialization,finished_mentions=sorted(finished),feedback=controller.verification_feedback,
            rng=rng.getstate(),exploration_steps=exploration,submitted=sorted(tried),result=result if done else None))
        append(folder/'progress.jsonl',dict(round=rnd,frontiers=len(fronts),archives=len(archives),
            unique_chains=len(chains),sufficient=result.get('evidence_complete',False)))
        if done:return result
    return result
