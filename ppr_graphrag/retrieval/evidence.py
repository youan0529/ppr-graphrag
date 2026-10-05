"""Archive compaction and checkpoint RNG restoration, preserved from r11."""

def tuple_tree(x):
    return tuple(tuple_tree(v) for v in x) if isinstance(x, list) else x


def compact_archives(archives):
    """Retire only exact prefix redundancy with covered local claims and no new source."""
    replacements = []
    active = [a for a in archives.values() if a.get("active", True)]
    def covers(new, old):
        return all(any(n["question"] == o["question"] and
            (n["status"] == o["status"] or n["status"] == "complete_local")
            for n in new["local_parts"]) for o in old["local_parts"])
    for old in sorted(active, key=lambda a: (len(a["path"]), a["id"])):
        choices = [new for new in active if new.get("active", True) and new["id"] != old["id"]
            and len(new["path"]) > len(old["path"])
            and new["path"][:len(old["path"])] == old["path"]
            and set(new["evidence"]["doc_ids"]) <= set(old["evidence"]["doc_ids"])
            and covers(new, old)]
        if choices and old["local_parts"]:
            new = min(choices, key=lambda a: (len(a["path"]), a["id"]))
            reason = "Exact prefix and all local claims covered; replacement adds no source document"
            old.update(active=False, replaced_by=new["id"], replacement_reason=reason)
            replacements.append(dict(old=old["id"], new=new["id"], reason=reason))
    return replacements
