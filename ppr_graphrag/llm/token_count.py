"""Token accounting for the pinned Ollama gpt-oss, text-only/no-tools template.

The template hash and token table are frozen with this experiment. This is not
a generic ChatGPT token estimator. Calls log both this count and server usage.
"""
import datetime
import json
from functools import lru_cache
from pathlib import Path

import tiktoken


@lru_cache(maxsize=1)
def encoding():
    base = tiktoken.get_encoding("o200k_base")
    special = json.loads((Path(__file__).resolve().parent / "harmony_special_tokens.json").read_text())
    return tiktoken.Encoding(name="pinned-gpt-oss-gguf", pat_str=base._pat_str,
                            mergeable_ranks=base._mergeable_ranks, special_tokens=special)


def render(messages, date=None):
    date = date or datetime.date.today().isoformat()
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rendered = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                f"Knowledge cutoff: 2024-06\nCurrent date: {date}\n\nReasoning: low\n\n"
                "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>")
    if system:
        rendered += "<|start|>developer<|message|>\n\n# Instructions\n\n" + system + "<|end|>"
    collated = []
    for message in messages:
        if message["role"] == "system":
            continue
        if message["role"] != "user":
            raise ValueError("Unaccounted message role in pinned template")
        if collated:
            collated[-1] += "\n\n" + message["content"]
        else:
            collated.append(message["content"])
    for content in collated:
        rendered += "<|start|>user<|message|>" + content + "<|end|>"
    return rendered + "<|start|>assistant"


def count(messages):
    return len(encoding().encode(render(messages), allowed_special="all"))


def response_count(messages, response):
    thinking = response.get("message", {}).get("thinking", "")
    if not thinking or response.get("done_reason") == "length":
        return count(messages)
    rendered = (render(messages) + "<|channel|>analysis<|message|>" + thinking
                + "<|end|><|start|>assistant<|channel|>final<|message|>")
    return len(encoding().encode(rendered, allowed_special="all"))
