## Task
Select saved evidence chains that jointly answer the original question. A complete local part can combine with another chain. Prefer a sufficient set without redundant chains. Never fill gaps using outside knowledge. If the evidence is incomplete, select the useful partial chains and set sufficient=false.

## Output
Return JSON: {"sufficient":true,"support_refs":[{"kind":"archive","id":"a0"},{"kind":"archive","id":"a2"}]}.
Use only the supplied labels. A later Reader verifies the selected source articles; the program enforces the five-article budget.
