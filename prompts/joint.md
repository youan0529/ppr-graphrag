## Task
Extract source-supported facts for a retrieval system that combines a directed entity graph with dense retrieval of factual statements. Start with subject–relation–object triples when the fact naturally has that form. We extend triples with qualifiers, endpoint semantic types, a routing kind. Do not force a statement into a triple by inventing an object or an entity.

The graph lets a reader move between identifiable entities and obtain terminal details about them. Dense retrieval preserves useful facts that cannot be reached this way. Extract the passage's informative facts, including relationships, aliases, dates, quantities and descriptions; choose their representation without adding outside knowledge.

## What can be a graph entity?
A graph entity denotes a particular, independently identifiable referent that can be recognized again in another fact. A named person, organization, place, work or event is usually suitable. An unambiguous mention can be resolved to its name from this passage. A rare name can still be an entity: do not estimate corpus frequency.

A local detail describes a value, role, class, quantity or situation without identifying a reusable referent. Examples include June 2011, 70 feature films, a 500-man force, 37 people, and an unresolved son or mother. These do not become graph nodes. Do not invent names for unnamed groups or events. A specifically identified regiment can be an entity, but merely calling an unnamed group a People group does not make it one. A named son can be an entity; the bare word son is not.

Decide this from what each endpoint refers to, not from its grammatical position or the kind of information the sentence provides. A particular named person, country, institution or film remains an entity when it is a birthplace, destination, employer or directed work. Thus a named director and a named film form ENTITY_LINK; a named person and a named country also form ENTITY_LINK. They do not become terminal details merely because the sentence describes the subject. The graph must allow a later question to continue through that named object.

Endpoint semantic types describe meaning. Routing kinds decide whether endpoints become graph nodes. These are different decisions: a detail can have type Person, People group, Role or occupation, or another appropriate type. Do not relabel a detail just to make it eligible as a node.

## Routing kinds
- ENTITY_LINK: both endpoints identify graph entities. The relation connects the subject entity to the object entity.
- ENTITY_TO_DETAIL: only the subject identifies a graph entity. The complete object detail is terminal evidence reached in the stated direction; it has no entity node or entity embedding. This includes more than simple numeric attributes.
- DETAIL_TO_ENTITY: only the object identifies a graph entity. Preserve the original triple. Also supply reverse_relation, a faithful predicate that reads from the original object entity to the original subject detail. This must express the same assertion with the endpoints exchanged, not an association or a guessed consequence. Use null if that cannot be done faithfully. Supply text, a standalone factual sentence preserving the assertion and its qualifiers; the program uses it for dense retrieval when reversal is unavailable.
- TEXT_FACT: neither endpoint is suitable, or the statement is not naturally a triple. Supply a standalone factual sentence instead of endpoints. It will be retrievable as text without creating graph nodes or invented graph hops. Include essential attribution, uncertainty, negation, quantities and temporal context.
- SAME_ENTITY: the passage explicitly establishes two names for the very same referent, such as a name and pen name or abbreviation. Both names denote graph entities. This is identity evidence, not an ordinary traversal step. Both endpoint types follow the referent: a bank's abbreviation is Organization, not Identifier or address. Descriptions, occupations and vague co-occurrence are not aliases.

Choose the representation of the actual assertion first. When the passage already supports a natural entity-to-detail formulation, use it directly rather than creating an unnecessary DETAIL_TO_ENTITY. Do not change who performed an action or what was appointed, measured or counted to obtain that direction. The program never traverses an incoming-only fact as though it were outgoing.

For each assertion, check explicit identity first, then identify its two endpoints: two reusable referents mean ENTITY_LINK; one reusable referent and one local detail mean one of the two directional detail kinds. Use TEXT_FACT only when a faithful entity-anchored triple is unavailable, not merely because the input is a complete sentence. A named film described as a 2008 documentary can take that description as a terminal detail. An explicitly named alias is SAME_ENTITY, not a terminal name string.

## Relations, details and qualifiers
Use a readable predicate with its original tense, polarity and role restrictions. Keep a detail sufficiently complete to retain its meaning. For "made 70 feature films", prefer relation "made" and detail "70 feature films"; neither "70" alone nor a generic film node preserves the fact. For "appointed him colonel of a 500-man force", preserve the appointment and the complete role "colonel of a 500-man force".

Qualifiers contain explicitly supported time, start_time, end_time, duration, location, condition or attribution. Keep essential meaning in the predicate/detail or standalone text; qualifiers supplement it. "In one afternoon" is duration, not location. Do not infer a named event, exact date or place from an article's topic. Resolve pronouns only when the passage makes the referent clear.

Read every resulting triple as a sentence: its subject must have the role asserted by the predicate. "Mira has a brother, Leo" supports "Mira has brother Leo", not "Mira is brother of Leo". A role is the detail itself, not a reason to repeat the person's name as an object. Standalone text must name its participants when the passage identifies them; do not leave a detached he, she or his. Omit unknown placeholders such as a question mark used for a missing birth date.

## Output
Return only {"facts":[...]} as JSON, at most 96 facts. Avoid duplicates.

For ENTITY_LINK, ENTITY_TO_DETAIL and SAME_ENTITY, use exactly:
{"kind":"...","subject":["mention","semantic type",""],"relation":"predicate","object":["mention or complete detail","semantic type",""],"qualifiers":{}}

For DETAIL_TO_ENTITY, use those fields plus:
"reverse_relation": "predicate from object back to subject, or null",
"text": "A standalone factual sentence including applicable qualifiers."
The actual unavailable reverse_relation value is JSON null, not the string "null".

For TEXT_FACT, use exactly:
{"kind":"TEXT_FACT","text":"A standalone source-supported factual sentence.","qualifiers":{}}

An endpoint is [text, semantic type, local disambiguator]. Use the type definitions below. The disambiguator is normally ""; use a short distinguishing phrase only for distinct referents with the same name within this passage, such as a novel and a film. Do not give local details an invented identity.

Input is a title followed by the original passage as plain text. The program records the source document for every fact. Do not output paragraph IDs, evidence fields, quotations or offsets.

## Complete example (synthetic passage)
Input:
Title: Sample biographical notes
Mira Vale was born in June 1970 in Alderford. She published under the pen name M. Reed. Her younger brother is Leo Vale.
Mira directed Northern Road, a 2008 documentary film. She studied at Alderford University, worked for Pine Studio, and visited Japan.
Between 1914 and 1958, Cecil Blount DeMille made 70 feature films.
The king appointed Sir Ralph Gore colonel of a 500-man force.
Five hundred unnamed soldiers served under Captain Rowan.
According to Bruce, 37 people were executed in one afternoon.
Banco de Crédito de Bolivia is abbreviated as BCP.
The play Sylvia was rerun in June 2011.

Output:
{"facts":[
{"kind":"ENTITY_TO_DETAIL","subject":["Mira Vale","Person",""],"relation":"was born in","object":["June 1970","Date or time",""],"qualifiers":{}},
{"kind":"ENTITY_LINK","subject":["Mira Vale","Person",""],"relation":"was born in","object":["Alderford","Place",""],"qualifiers":{"time":"June 1970"}},
{"kind":"SAME_ENTITY","subject":["Mira Vale","Person",""],"relation":"published under the pen name","object":["M. Reed","Person",""],"qualifiers":{}},
{"kind":"ENTITY_LINK","subject":["Mira Vale","Person",""],"relation":"has younger brother","object":["Leo Vale","Person",""],"qualifiers":{}},
{"kind":"ENTITY_LINK","subject":["Mira Vale","Person",""],"relation":"directed","object":["Northern Road","Creative work",""],"qualifiers":{}},
{"kind":"ENTITY_TO_DETAIL","subject":["Northern Road","Creative work",""],"relation":"is a","object":["2008 documentary film","Descriptive value",""],"qualifiers":{}},
{"kind":"ENTITY_LINK","subject":["Mira Vale","Person",""],"relation":"studied at","object":["Alderford University","Organization",""],"qualifiers":{}},
{"kind":"ENTITY_LINK","subject":["Mira Vale","Person",""],"relation":"worked for","object":["Pine Studio","Organization",""],"qualifiers":{}},
{"kind":"ENTITY_LINK","subject":["Mira Vale","Person",""],"relation":"visited","object":["Japan","Place",""],"qualifiers":{}},
{"kind":"ENTITY_TO_DETAIL","subject":["Cecil Blount DeMille","Person",""],"relation":"made","object":["70 feature films","Number or quantity",""],"qualifiers":{"start_time":"1914","end_time":"1958"}},
{"kind":"ENTITY_TO_DETAIL","subject":["Sir Ralph Gore","Person",""],"relation":"was appointed","object":["colonel of a 500-man force","Role or occupation",""],"qualifiers":{}},
{"kind":"DETAIL_TO_ENTITY","subject":["Five hundred unnamed soldiers","People group",""],"relation":"served under","object":["Captain Rowan","Person",""],"qualifiers":{},"reverse_relation":"had serving under them","text":"Five hundred unnamed soldiers served under Captain Rowan."},
{"kind":"TEXT_FACT","text":"According to Bruce, 37 people were executed in one afternoon.","qualifiers":{"attribution":"Bruce","duration":"one afternoon"}},
{"kind":"SAME_ENTITY","subject":["Banco de Crédito de Bolivia","Organization",""],"relation":"is abbreviated as","object":["BCP","Organization",""],"qualifiers":{}},
{"kind":"ENTITY_TO_DETAIL","subject":["Sylvia","Creative work",""],"relation":"was rerun in","object":["June 2011","Date or time",""],"qualifiers":{}}
]}

## Endpoint semantic types
Person: a real person, including names or aliases referring to that person.
Fictional character: an individual explicitly presented as a fictional character.
Organization: an institution, company or organized body, such as a school or team.
People group: people sharing an identity or condition, without an organizational identity.
Place: a geographic area, administrative area, building or facility as a location.
Physical object: a tangible object, device, body part or item model; natural or manufactured.
Organism: a non-human, non-fictional organism or kind of organism, including animals, plants, fungi and microorganisms.
Substance: a material, chemical element, compound or mixture, rather than a discrete item.
Creative work: a literary, film, television, musical or artistic work.
Document: an informational or official text, such as a paper, report or law.
Software: an application, operating system, program or software library.
Dataset: a collection of data or records referred to as a whole.
Service: a service being offered, rather than its provider or one delivery of it.
Event: a specific occurrence, gathering or activity, including planned ones; an overall project or program is separate.
Project or program: an organized plan or coordinated undertaking pursuing an objective, not its written document or one constituent event.
Role or occupation: a social role or title, job position or profession, not the person holding it.
Language: a natural, signed or programming language.
Award: an award, medal or honor itself, not its recipient, ceremony or physical prize.
Academic field: a discipline or field of study.
Method: a general method, algorithm, technique or procedure, not a particular project or one execution.
Religion or ideology: a religion, doctrine or ideological system, not its followers, organizations or ordinary personal opinions.
Genre: a literary, film, musical or artistic genre or style.
Disease or symptom: a disease, disorder, syndrome, injury, symptom or clinical sign.
Date or time: a calendar date, year, month, time of day, combined date and time, or interval locating when something occurs; not an elapsed duration.
Duration: an elapsed or required length of time.
Number or quantity: a count, rank, numerical value, measurement, proportion or monetary amount; dates, durations and identifiers are separate.
Truth value: an explicitly stated yes/no or true/false value.
Identifier or address: a code, serial number, version label or web address used as an identifier or address, even if purely numeric.
Descriptive value: a descriptive property value, such as a color, state, nationality adjective or evaluation, without a more specific listed category.
Other: the endpoint's meaning is clear, but no listed category fits.
Unknown: the context does not establish what the endpoint denotes.
