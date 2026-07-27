"""Exact match and token F1 for short-answer QA."""

from __future__ import annotations

from collections import Counter
import re
import string


SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    normalized = normalize_answer(prediction)
    return max((float(normalized == normalize_answer(answer)) for answer in answers), default=0.0)


def token_f1(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()
    scores = []
    for answer in answers:
        normalized_answer = normalize_answer(answer)
        answer_tokens = normalized_answer.split()
        if (
            (normalized_prediction in SPECIAL_ANSWERS or normalized_answer in SPECIAL_ANSWERS)
            and normalized_prediction != normalized_answer
        ):
            scores.append(0.0)
            continue
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        overlap = sum(common.values())
        if not prediction_tokens or not answer_tokens:
            scores.append(float(prediction_tokens == answer_tokens))
            continue
        if overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)
