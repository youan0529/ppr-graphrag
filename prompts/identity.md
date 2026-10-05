## Task
Decide whether two named graph entities denote the same real-world entity. They are only candidates selected by vector similarity. Similar names, related topics, a family relationship, common geography or the same type do not establish identity.

Read the names, article titles and local facts. Return same only when the available context gives a credible identity match, such as an unambiguous shortened name or an explicitly supported alias. Return different for distinct people, organizations, works or places. Return unknown if the context cannot distinguish identity from similarity. Do not invent an alias from vector similarity.

Parent and child remain different people even when they share most of a name or title. A company and its subsidiary remain different organizations. A country and its capital are different places. An actor and a character are different. Differences in dates, ordinal titles or incompatible roles are evidence against identity. The local facts may contain extraction errors; resolve uncertainty as unknown rather than explaining it away.

## Complete example
Input:
Pair 0
Left: Edward Hall, 1st Baron [Person], article Edward Hall, 2nd Baron
Local facts: Edward Hall, 2nd Baron — son of → Edward Hall, 1st Baron
Right: Edward Hall, 2nd Baron [Person], article Edward Hall, 2nd Baron
Local facts: Edward Hall, 2nd Baron — son of → Edward Hall, 1st Baron

Pair 1
Left: Mira Chen [Person], article Mira Chen
Local facts: Mira Chen — directed → Blue Harbour
Right: Chen [Person], article Blue Harbour
Local facts: Blue Harbour — directed by → Chen; Chen — also known as → Mira Chen

Output:
{"pairs":[
{"id":0,"reason":"The ordinal titles and explicit father/son relation identify two people.","decision":"different"},
{"id":1,"reason":"The supplied alias and matching film identify the same director.","decision":"same"}
]}

## Output
Return JSON with one id, short reason and decision (same, different or unknown) for every pair. There is no requirement to merge any pair.
