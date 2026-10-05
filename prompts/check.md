## Task
Look for evidence-chain prefixes that jointly answer the original question. Combine newly sampled chains with saved evidence. A later Reader verifies the source articles. If the needed evidence is present, submit it now; do not wait for more exploration merely because the current chain was pursuing a local part.

## Reading and pruning
Each step reads subject → relation → object in its displayed direction, using the actual names in that fact. Linked names connect adjacent steps through a graph identity hypothesis; the source title helps identify the referent. The Reader will verify these identities against the articles. Judge whether the supplied chains provide a plausible answer; do not fill missing facts from outside knowledge.
Keep useful prefixes even when later steps are irrelevant. Cut an unnecessary tail before saving or submitting. Preserve connecting steps, not detached values. One completed side of a comparison should be saved while the other side is explored.
Position 0 is the starting entity. Position i is the endpoint of step i. A terminal detail can answer a question but cannot be extended. To seek another property, go back to the entity BEFORE that detail.

## Decision
Return submit if some prefixes and saved chains jointly provide all needed relationships and properties. Select a sufficient set without redundant tails. The program enforces the document budget.
Otherwise return explore. In updates, give one entry keyed by the supplied short label for every NEW chain. keep_to is the last useful step, or 0 if none. restart_at is an entity position to extend, or null to stop this chain. A saved prefix and a continuation can coexist. Use keep_to>0 and restart_at=null to freeze useful evidence. Use keep_to=0 and restart_at=null to discard an unhelpful chain.
relations gives a few shared X/Y relation hints for what is still missing, without guessed answers. finished_mentions lists named starting anchors whose entire contribution needed by the original question is now saved; these release their exploration slots. Finding an intermediate participant does not finish its task if a required property is still missing. Use [] when uncertain.

## Output
For submission:
{"decision":"submit","support_refs":[{"kind":"archive","id":"a0"},{"kind":"path","id":"p1","end_at":2}]}
For exploration:
{"decision":"explore","updates":{"p0":{"keep_to":2,"restart_at":null},"p1":{"keep_to":1,"restart_at":1},"p2":{"keep_to":0,"restart_at":null}},"relations":["X was born in Y"],"finished_mentions":["Northern Road"]}
The three updates above refer to p0, p1 and p2. Never omit an entry. Return only JSON.

## Example
Original question: Which film's director was born earlier, Northern Road or Summer House?
Named starting anchors: Northern Road; Summer House
New chain p0:
  1. Northern Road → directed by → Mira Vale
  2. Mira Vale → born in → 1960 [terminal detail]
New chain p1:
  1. Summer House → directed by → Leo Reed
  2. Leo Reed → married to → Nora Reed
New chain p2:
  1. Summer House → produced by → Pine Studio
  2. Pine Studio → based in → Alderford
Output:
{"decision":"explore","updates":{"p0":{"keep_to":2,"restart_at":null},"p1":{"keep_to":1,"restart_at":1},"p2":{"keep_to":0,"restart_at":null}},"relations":["X was born in Y"],"finished_mentions":["Northern Road"]}
