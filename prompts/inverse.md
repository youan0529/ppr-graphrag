## Task
Give a useful reverse relation for each numbered group of triples. We use these labels to walk through a graph in the reverse direction. The original facts and source articles remain available for later reasoning.

## Direction
For an input A → relation → B, return the reverse_relation part of B → [reverse_relation] → A. For example, OTRAG → based in → Stuttgart asks for the reverse_relation part of Stuttgart → [reverse_relation] → OTRAG.

Prefer a short, familiar phrase that clearly expresses who does what to whom in this reversed order. Paraphrase and generalize when useful: is mother of can become is child of; is son of can become is parent of, without guessing the other person's gender. For a place, is location of or contains can both be useful. You do not need the original wording, strict semantic equivalence, or every detail.

Keep the roles and direction correct. A character was portrayed by an actor; it did not portray the actor. If a preposition is needed to express the direction, include it in reverse_relation, for example by or of. Do not omit a necessary preposition just to shorten the label. Preserve negation. A triple need not form a complete sentence, but use a role name or fragment only when a clear conventional phrase is hard to express. If no useful reverse wording is possible, return null; the fact remains available through the forward edge and dense retrieval.

## Complete example
Input:
Relation 0: OTRAG [Organization] → based in → Stuttgart [Place]
Relation 1: Blue Harbour [Creative work] → stars → Mira Chen [Person]
Relation 2: Alex [Person] → played → Officer Lee [Fictional character]
Relation 3: Leon [Person] → was emperor of → Arcadia [Place]
Relation 4: Morgan [Person] → was MP for → West County [Place]

Output:
{"inverses":[
{"id":0,"reverse_relation":"is location of"},
{"id":1,"reverse_relation":"starred in"},
{"id":2,"reverse_relation":"was portrayed by"},
{"id":3,"reverse_relation":"had an emperor"},
{"id":4,"reverse_relation":"was represented by"}
]}

## Output
Return JSON with inverses. Each row contains only the given id and reverse_relation (a short label or JSON null). A numbered group may include additional triples showing the same predicate with other endpoint types; return ONE label for that group. Keep the given IDs, including when they are not consecutive. Do not include endpoint names or placeholders in the label. Do not repeat the original relation or add an explanation; the program retains the facts and matches labels by id.
