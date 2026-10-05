"""Compact model outputs; IDs and provenance are supplied by the program."""


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array(items, maximum=64):
    return {"type": "array", "items": items, "maxItems": maximum}


def record(items):
    # Ollama's installed grammar converter supports draft-07 tuple items, not boolean items.
    return {"type": "array", "items": items, "minItems": len(items), "maxItems": len(items)}


STRING = {"type": "string"}
TYPES = [
    "PERSON",
    "ORG",
    "NORP",
    "GPE",
    "LOC",
    "FAC",
    "WORK_OF_ART",
    "PRODUCT",
    "EVENT",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "MONEY",
    "PERCENT",
    "CHARACTER",
    "OTHER",
]
VALUE_TYPES = {"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"}
KINDS = ["ENTITY_RELATION", "ATTRIBUTE", "IDENTITY"]
QUALIFIER_KEYS = ["time", "start_time", "end_time", "location", "condition"]
FACT_STRING = {"type": "string", "minLength": 1}
QUALIFIERS = {"type": "object", "properties": {k: FACT_STRING for k in QUALIFIER_KEYS}, "additionalProperties": False}
NER = obj({"mentions": array(record([STRING, {"enum": TYPES}]), 64)})
EXTRACTION = obj({"facts": array(record([FACT_STRING, FACT_STRING, FACT_STRING, QUALIFIERS, {"enum": KINDS}]), 64)})
READER = obj(
    {
        "chain_verdict": {"enum": ["valid", "corrected", "invalid", "insufficient"]},
        "evidence_sufficient": {"type": "boolean"},
        "reason": STRING,
        "answer": STRING,
        "reopen_chain_ids": array(STRING),
    }
)
