"""Thinking Quality Scorer: evaluate chain-of-thought reasoning trace quality.

Scores a model's <think> trace on logical coherence, step structure, repetition,
conclusion presence, and relevance. Used as a GRPO reward component to train the
model toward grounded, structured reasoning rather than copy-paste or rambling.

For a TS dev: like a linter for reasoning — it checks whether the thinking trace
has the right "shape" (steps, conclusion, no repetition) and is logically sound.

Architecture:
    trace ──► ThinkingQualityScorer ──► ThinkingScore (0-1 per dimension)

Self-contained: no external dependencies beyond the standard library.
"""

import re
from dataclasses import dataclass, field
from collections import Counter

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThinkingScore:
    """Quality score for a chain-of-thought reasoning trace.

    All component scores are in [0, 1]. The overall score is a weighted
    combination of the components.
    """

    # Weighted composite
    overall: float = 0.0

    # Component scores (0-1)
    coherence: float = 0.0      # Does the trace follow a logical progression?
    repetition: float = 0.0     # Absence-of-repetition score (1 = no repetition)
    conclusion: float = 0.0     # Does the trace reach a clear conclusion?
    step_quality: float = 0.0   # Are there an appropriate number of clear steps?

    # Free-form details for debugging / logging
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Component weights
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "coherence":    0.30,
    "repetition":   0.25,
    "conclusion":   0.25,
    "step_quality": 0.20,
}

# ---------------------------------------------------------------------------
# Conclusion signal patterns
# ---------------------------------------------------------------------------

_CONCLUSION_PATTERNS = [
    r"\btherefore\b",
    r"\bthus\b",
    r"\bin conclusion\b",
    r"\bso\b,?\s+i\s+will\b",
    r"\bso\s+the\s+answer\b",
    r"\bso\s+we\b",
    r"\bfinally\b",
    r"\bmy\s+approach\b",
    r"\bi\s+will\s+implement\b",
    r"\bi\s+will\s+use\b",
    r"\bthe\s+solution\s+is\b",
    r"\bwe\s+can\s+(?:now\s+)?(?:write|implement|use)\b",
    r"\bto\s+summarize\b",
    r"\bin\s+summary\b",
    r"\bthe\s+final\b",
    r"\boverall\b",
]

# ---------------------------------------------------------------------------
# Coherence transition signals
# ---------------------------------------------------------------------------

_TRANSITION_SIGNALS = [
    r"\bfirst\b",
    r"\bsecond\b",
    r"\bthird\b",
    r"\bnext\b",
    r"\bthen\b",
    r"\bafter(?:ward)?\b",
    r"\bfinally\b",
    r"\bsince\b",
    r"\bbecause\b",
    r"\bhowever\b",
    r"\balternatively\b",
    r"\bthis\s+means\b",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bgiven\s+that\b",
    r"\bwe\s+need\b",
    r"\bwe\s+should\b",
    r"\bwe\s+can\b",
    r"\bstep\s+\d+\b",
]


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class ThinkingQualityScorer:
    """Score the quality of a chain-of-thought reasoning trace.

    Usage:
        scorer = ThinkingQualityScorer()
        result = scorer.score(trace)
        print(result.overall)   # 0.0 – 1.0
    """

    def score(self, thinking_trace: str) -> ThinkingScore:
        """Compute a composite quality score for the given reasoning trace.

        Args:
            thinking_trace: Raw text of the chain-of-thought trace.

        Returns:
            ThinkingScore with per-dimension and overall scores.
        """
        if not thinking_trace or not thinking_trace.strip():
            return ThinkingScore(
                overall=0.0,
                coherence=0.0,
                repetition=0.0,
                conclusion=0.0,
                step_quality=0.0,
                details={"error": "empty trace"},
            )

        coherence_score, coherence_details = self.check_coherence(thinking_trace)
        repetition_score, repetition_details = self.check_repetition(thinking_trace)
        conclusion_score, conclusion_details = self.check_conclusion(thinking_trace)
        step_score, step_details = self.check_step_count(thinking_trace)

        overall = (
            _WEIGHTS["coherence"]    * coherence_score +
            _WEIGHTS["repetition"]   * repetition_score +
            _WEIGHTS["conclusion"]   * conclusion_score +
            _WEIGHTS["step_quality"] * step_score
        )
        overall = max(0.0, min(1.0, overall))

        return ThinkingScore(
            overall=overall,
            coherence=coherence_score,
            repetition=repetition_score,
            conclusion=conclusion_score,
            step_quality=step_score,
            details={
                "coherence":    coherence_details,
                "repetition":   repetition_details,
                "conclusion":   conclusion_details,
                "step_quality": step_details,
            },
        )

    # ------------------------------------------------------------------
    # check_coherence
    # ------------------------------------------------------------------

    def check_coherence(self, trace: str) -> tuple[float, dict]:
        """Check whether the trace follows a logical progression.

        Heuristics:
        - Presence of transition / logical connective words ("first", "then",
          "therefore", "because", "since", …)
        - Average sentence length is in a reasonable range (not single-word
          fragments or run-on walls of text)
        - The trace has at least 2 sentences / lines of substance

        Returns:
            (score 0-1, details dict)
        """
        trace_lower = trace.lower()
        sentences = _split_sentences(trace)

        if not sentences:
            return 0.0, {"reason": "no sentences found"}

        # 1. Transition signal density
        signal_hits = sum(
            1
            for pattern in _TRANSITION_SIGNALS
            if re.search(pattern, trace_lower)
        )
        # Normalise: hitting >=5 distinct signals is considered full score
        transition_score = min(1.0, signal_hits / 5.0)

        # 2. Sentence length health: penalise if average is very short (< 4 words)
        #    or very long (> 60 words — likely a run-on / prompt dump)
        word_counts = [len(s.split()) for s in sentences if s.strip()]
        if not word_counts:
            avg_words = 0.0
        else:
            avg_words = sum(word_counts) / len(word_counts)

        if avg_words < 4:
            length_score = 0.2
        elif avg_words > 60:
            length_score = 0.4
        else:
            # Map 4-60 to 0.5-1.0 with a peak around 10-20 words
            length_score = min(1.0, 0.5 + (min(avg_words, 20) - 4) / 32.0)

        # 3. Minimum substance: at least 2 non-trivial lines
        substance_lines = [s for s in sentences if len(s.split()) >= 4]
        substance_score = min(1.0, len(substance_lines) / 3.0)

        coherence = 0.40 * transition_score + 0.35 * length_score + 0.25 * substance_score

        details = {
            "transition_signals_hit": signal_hits,
            "transition_score": round(transition_score, 3),
            "avg_words_per_sentence": round(avg_words, 1),
            "length_score": round(length_score, 3),
            "substance_lines": len(substance_lines),
            "substance_score": round(substance_score, 3),
        }
        return round(coherence, 4), details

    # ------------------------------------------------------------------
    # check_repetition
    # ------------------------------------------------------------------

    def check_repetition(self, trace: str) -> tuple[float, dict]:
        """Detect repeated reasoning in the trace.

        Heuristics:
        - Bigram / trigram repetition ratio: high overlap signals copy-paste
          or looping reasoning
        - Exact duplicate lines ratio
        - Sentence-level near-duplicate ratio (Jaccard similarity > 0.8)

        Returns:
            (score 0-1, where 1 = no repetition, 0 = highly repetitive)
        """
        lines = [ln.strip() for ln in trace.splitlines() if ln.strip()]

        if not lines:
            return 1.0, {"reason": "no content"}

        # 1. Exact duplicate lines
        line_counter = Counter(lines)
        duplicate_lines = sum(count - 1 for count in line_counter.values() if count > 1)
        duplicate_ratio = duplicate_lines / len(lines)

        # 2. Trigram repetition
        words = re.findall(r'\b\w+\b', trace.lower())
        trigram_rep_ratio = 0.0
        if len(words) >= 6:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            trigram_counter = Counter(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counter.values() if c > 1)
            trigram_rep_ratio = repeated_trigrams / len(trigrams)

        # 3. Near-duplicate sentence pairs (Jaccard)
        sentences = _split_sentences(trace)
        near_dup_pairs = 0
        total_pairs = 0
        if len(sentences) >= 2:
            tokenised = [set(re.findall(r'\b\w+\b', s.lower())) for s in sentences]
            for i in range(len(tokenised)):
                for j in range(i + 1, len(tokenised)):
                    a, b = tokenised[i], tokenised[j]
                    if not a or not b:
                        continue
                    total_pairs += 1
                    jaccard = len(a & b) / len(a | b)
                    if jaccard > 0.8:
                        near_dup_pairs += 1
        near_dup_ratio = (near_dup_pairs / total_pairs) if total_pairs > 0 else 0.0

        # Combine: weight duplicate lines most heavily
        penalty = (
            0.45 * duplicate_ratio +
            0.30 * trigram_rep_ratio +
            0.25 * near_dup_ratio
        )
        # score = 1 - penalty, clamped
        score = max(0.0, 1.0 - penalty * 2.5)  # scale up so moderate repetition hurts

        details = {
            "duplicate_line_ratio": round(duplicate_ratio, 3),
            "trigram_repetition_ratio": round(trigram_rep_ratio, 3),
            "near_duplicate_sentence_ratio": round(near_dup_ratio, 3),
            "combined_penalty": round(penalty, 3),
        }
        return round(score, 4), details

    # ------------------------------------------------------------------
    # check_conclusion
    # ------------------------------------------------------------------

    def check_conclusion(self, trace: str) -> tuple[float, dict]:
        """Check whether the trace reaches a clear conclusion.

        Heuristics:
        - Presence of conclusion signal words in the final 30% of the trace
        - Whether the last substantive sentence is longer / more specific
          than the rest (conclusions tend to be more precise)

        Returns:
            (score 0-1, details dict)
        """
        trace_lower = trace.lower()

        # 1. Any conclusion signal anywhere
        global_hits = [p for p in _CONCLUSION_PATTERNS if re.search(p, trace_lower)]

        # 2. Conclusion signal in the final third
        final_portion = trace_lower[int(len(trace_lower) * 0.65):]
        final_hits = [p for p in _CONCLUSION_PATTERNS if re.search(p, final_portion)]

        # 3. Last substantive sentence heuristic
        sentences = [s for s in _split_sentences(trace) if len(s.split()) >= 4]
        last_sentence_bonus = 0.0
        if sentences:
            last = sentences[-1].lower()
            # Bonus if last sentence contains an action verb ("implement", "use",
            # "return", "call", "write") — signals a concrete decision
            action_verbs = [
                r"\bimplement\b", r"\buse\b", r"\bwrite\b", r"\breturn\b",
                r"\bcall\b", r"\bcreate\b", r"\bapply\b", r"\bbuild\b",
            ]
            if any(re.search(v, last) for v in action_verbs):
                last_sentence_bonus = 0.25

        global_score = min(1.0, len(global_hits) / 2.0)
        final_score  = min(1.0, len(final_hits) / 1.5)

        conclusion = 0.35 * global_score + 0.40 * final_score + 0.25 * last_sentence_bonus
        conclusion = min(1.0, conclusion)

        details = {
            "global_signal_count": len(global_hits),
            "final_portion_signal_count": len(final_hits),
            "last_sentence_action_verb": last_sentence_bonus > 0,
            "global_score": round(global_score, 3),
            "final_score": round(final_score, 3),
        }
        return round(conclusion, 4), details

    # ------------------------------------------------------------------
    # check_step_count
    # ------------------------------------------------------------------

    def check_step_count(self, trace: str) -> tuple[float, dict]:
        """Check whether the trace has an appropriate number of reasoning steps.

        "Appropriate" is defined as 3–8 steps for most coding tasks. Fewer
        suggests underthinking; more suggests excessive verbosity or a very
        hard problem (slight penalty only for extreme cases).

        Step detection:
        - Explicit "Step N:" / "N." / "N)" numbered labels
        - Sentence-level count when no explicit numbering is present

        Returns:
            (score 0-1, details dict)
        """
        # Count explicit numbered steps
        explicit_steps = len(re.findall(
            r'(?:^|\n)\s*(?:step\s+\d+|^\d+[.)]\s)',
            trace,
            re.IGNORECASE | re.MULTILINE,
        ))

        sentences = [s for s in _split_sentences(trace) if len(s.split()) >= 5]
        sentence_steps = len(sentences)

        # Choose the more meaningful count
        if explicit_steps >= 2:
            step_count = explicit_steps
            method = "explicit"
        else:
            step_count = sentence_steps
            method = "sentence"

        # Scoring curve: ideal is 3-8
        if step_count == 0:
            score = 0.0
        elif step_count == 1:
            score = 0.3
        elif step_count == 2:
            score = 0.6
        elif 3 <= step_count <= 8:
            score = 1.0
        elif 9 <= step_count <= 12:
            # Slightly long but acceptable
            score = 0.8
        elif 13 <= step_count <= 20:
            score = 0.6
        else:
            # Very long traces — probably rambling
            score = max(0.3, 1.0 - (step_count - 20) * 0.03)

        details = {
            "explicit_step_count": explicit_steps,
            "sentence_step_count": sentence_steps,
            "step_count_used": step_count,
            "count_method": method,
        }
        return round(score, 4), details


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def score_reasoning(trace: str) -> float:
    """Convenience wrapper — returns the overall quality score (0-1).

    Args:
        trace: Chain-of-thought reasoning trace text.

    Returns:
        Float in [0, 1] representing overall reasoning quality.
    """
    scorer = ThinkingQualityScorer()
    return scorer.score(trace).overall


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?', or newlines.

    Returns a list of stripped sentence strings with at least one word.
    """
    # Split on sentence-ending punctuation or newlines
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]
