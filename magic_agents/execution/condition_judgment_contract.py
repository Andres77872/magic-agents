"""Validate the Jev JSON answer contract shared by all judgment providers.

Preserve native Jev values. General LLM probabilities are normalized and adapted
before this validation; their confidence estimates are never calibrated.
"""
from __future__ import annotations

import math
from typing import Any


def probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Probabilities must be numbers, not strings or booleans')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Probabilities must be finite numbers between 0 and 1')
    return float(value)


def validate_judgment_answers(raw: Any, questions: dict[str, dict]) -> dict[str, dict]:
    """Reject partial, mismatched, ambiguous or nonfinite judgments atomically."""
    if not isinstance(raw, dict) or set(raw) != set(questions):
        raise ValueError('Response question IDs must exactly match configured questions')
    results = {}
    for key, question in questions.items():
        answer = raw[key]
        kind = question['type']
        fields = {'type', 'noul'} if kind == 'noul' else (
            {'type', 'choice', 'probabilities', 'confidence'} if kind == 'choice' else
            {'type', 'score', 'legend', 'probabilities', 'confidence'}
        )
        if not isinstance(answer, dict) or set(answer) != fields or answer.get('type') != kind:
            raise ValueError('Response answer fields must match the configured question type')
        if kind == 'noul':
            results[key] = {'type': kind, 'noul': probability(answer['noul'])}
            continue
        labels = list(question['criteria']) if kind == 'choice' else [str(i) for i in range(len(question['criteria']))]
        raw_probabilities = answer['probabilities']
        if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != set(labels):
            raise ValueError('Probability labels must exactly match the configured criteria')
        probabilities = {label: probability(raw_probabilities[label]) for label in labels}
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-4):
            raise ValueError('Judgment probabilities must sum to one')
        confidence = probability(answer['confidence'])
        if kind == 'choice':
            choice = answer['choice']
            if not isinstance(choice, str) or choice not in labels:
                raise ValueError('Choice must be one of the configured criterion labels')
            if not math.isclose(probabilities[choice], max(probabilities.values()), abs_tol=1e-6):
                raise ValueError('Choice must have the highest probability')
            results[key] = {'type': kind, 'choice': choice, 'probabilities': probabilities, 'confidence': confidence}
        else:
            score = answer['score']
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= len(labels) - 1):
                raise ValueError('Score must be a finite number within the configured scale')
            expected = sum(i * probabilities[str(i)] for i in range(len(labels)))
            if not math.isclose(score, expected, abs_tol=1e-4):
                raise ValueError('Score must equal the probability-weighted criterion index')
            legend = {str(i): description for i, description in enumerate(question['criteria'])}
            if answer['legend'] != legend:
                raise ValueError('Score legend must match the configured criterion descriptions')
            results[key] = {'type': kind, 'score': float(score), 'legend': legend,
                            'probabilities': probabilities, 'confidence': confidence}
    return results
