## Task
Answer the question using only the supplied source articles. Candidate chains help locate the reasoning; they may contain incorrect graph merges, extra steps or extraction errors. Read the articles and establish the needed participants, relationships and properties. You may correct a candidate chain when these articles support the answer. Return the shortest source-supported answer, normally a name, date, place, or yes/no, without explanation in the answer field.

## Verification
Use valid when the selected chains and source articles support the answer. Use corrected when the articles support the answer through a corrected interpretation of the candidate chains. Both allow an answer. Use insufficient for missing necessary evidence and invalid for unsupported reasoning without a source-supported repair; these require answer="unknown". Do not reject a supported answer merely because an unused extra chain is bad. Check direction, identity, participant roles, negation and conditions against the articles. An embedding identity merge is a hypothesis, not evidence.
In reopen_chain_ids name only supplied chains whose needed evidence requires more exploration. Preserve correctly supported parts. Give a brief reason, with missing evidence when applicable. Keep the answer aligned with the requested relation: a job title, an employer, and a geographic workplace are different properties. Prefer an explicit relationship in the articles over an institution inferred merely from a job title. Do not add a current-day restriction unless the question explicitly requests the current situation. Preserve the spelling and punctuation of source names. Do not use external knowledge.

## Output
{"chain_verdict":"valid|corrected|invalid|insufficient","evidence_sufficient":true,"reason":"Brief source-based explanation","answer":"Short answer","reopen_chain_ids":[]}
valid/corrected require evidence_sufficient=true. invalid/insufficient require false and answer="unknown". valid requires an empty reopen_chain_ids.

## Example
Question: Which film's director was born earlier, Northern Road or Summer House?
Articles state that Northern Road was directed by Mira Vale, born in 1960, and Summer House was directed by Leo Reed, born in 1970.
Output:
{"chain_verdict":"valid","evidence_sufficient":true,"reason":"The articles identify both directors; 1960 precedes 1970.","answer":"Northern Road","reopen_chain_ids":[]}
