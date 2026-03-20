"""Confidence-Based Routing: route to specialists with confidence checks.

Extends the basic router with:
- Confidence thresholds per domain
- Fallback to general model when confidence is low
- Cascade routing: specialist → general → ensemble
- Logging and analytics of routing decisions
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class RoutingDecision:
    """Record of a single routing decision."""
    timestamp: float
    input_preview: str  # First 100 chars of input
    domain: str
    confidence: float
    method: str  # "model", "heuristic", "fallback", "cascade"
    specialist_used: str | None  # Actual specialist that handled it
    latency_ms: float


class ConfidenceRouter:
    """Routes requests to specialists based on confidence scores.

    Routing strategy:
    1. Run domain detection (model or heuristic)
    2. If confidence >= domain threshold → use specialist
    3. If confidence < threshold → fall back to general model
    4. Optional cascade: try specialist, check output quality, re-route if bad
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        domain_thresholds: dict[str, float] | None = None,
        enable_cascade: bool = False,
        log_decisions: bool = True,
        log_path: str = "data/routing_log.jsonl",
    ):
        """
        Args:
            default_threshold: Default confidence threshold for routing.
            domain_thresholds: Per-domain confidence thresholds.
            enable_cascade: Enable cascade routing (try specialist, fall back if poor quality).
            log_decisions: Whether to log routing decisions.
            log_path: Path to routing decision log file.
        """
        self.default_threshold = default_threshold
        self.domain_thresholds = domain_thresholds or {}
        self.enable_cascade = enable_cascade
        self.log_decisions = log_decisions
        self.log_path = Path(log_path)

        # Analytics
        self.decisions: list[RoutingDecision] = []
        self.domain_counts: dict[str, int] = defaultdict(int)
        self.fallback_count: int = 0
        self.cascade_count: int = 0

    def get_threshold(self, domain: str) -> float:
        """Get confidence threshold for a domain."""
        return self.domain_thresholds.get(domain, self.default_threshold)

    def route(self, code: str, domain_scores: list | None = None) -> RoutingDecision:
        """Make a routing decision for a code snippet.

        Args:
            code: The input code/prompt.
            domain_scores: Pre-computed domain scores (optional, will compute if not provided).

        Returns:
            RoutingDecision with domain and routing method.
        """
        start = time.time()

        # Get domain scores
        if domain_scores is None:
            from cola_coder.features.domain_detector import detect_domain
            domain_scores = detect_domain(code)

        # Check top domain confidence against threshold
        if domain_scores and len(domain_scores) > 0:
            top_domain = domain_scores[0].domain
            top_confidence = domain_scores[0].confidence
            threshold = self.get_threshold(top_domain)

            if top_confidence >= threshold:
                method = "model" if hasattr(domain_scores[0], '_from_model') else "heuristic"
                specialist = top_domain
            else:
                method = "fallback"
                specialist = "general"
                top_domain = "general"
                top_confidence = 1.0 - sum(s.confidence for s in domain_scores[:3])
                self.fallback_count += 1
        else:
            top_domain = "general"
            top_confidence = 1.0
            method = "fallback"
            specialist = "general"
            self.fallback_count += 1

        latency = (time.time() - start) * 1000  # ms

        decision = RoutingDecision(
            timestamp=time.time(),
            input_preview=code[:100].replace('\n', ' '),
            domain=top_domain,
            confidence=top_confidence,
            method=method,
            specialist_used=specialist,
            latency_ms=latency,
        )

        self.decisions.append(decision)
        self.domain_counts[top_domain] += 1

        # Log to file
        if self.log_decisions:
            self._log_decision(decision)

        return decision

    def route_with_cascade(self, code: str, generators: dict = None) -> RoutingDecision:
        """Route with cascade: try specialist, fall back if quality is poor.

        Args:
            code: Input code/prompt.
            generators: Dict mapping domain -> generator. If None, basic routing only.

        Returns:
            RoutingDecision.
        """
        # First routing decision
        decision = self.route(code)

        if not self.enable_cascade or generators is None:
            return decision

        # If we have generators, try specialist and measure quality
        specialist_domain = decision.domain
        if specialist_domain in generators and specialist_domain != "general":
            try:
                output = generators[specialist_domain].generate(
                    prompt=code, max_new_tokens=50, temperature=0.3
                )

                # Quick quality check: is the output reasonable?
                if self._check_output_quality(output):
                    return decision  # Specialist output is good

                # Fall back to general
                decision.method = "cascade"
                decision.specialist_used = "general"
                decision.domain = "general"
                self.cascade_count += 1

            except Exception:
                decision.method = "cascade"
                decision.specialist_used = "general"
                decision.domain = "general"
                self.cascade_count += 1

        return decision

    def _check_output_quality(self, output: str) -> bool:
        """Quick quality check on generated output."""
        if not output or len(output.strip()) < 5:
            return False

        # Check for repetition
        from cola_coder.features.smoke_test import detect_repetition
        if detect_repetition(output):
            return False

        return True

    def _log_decision(self, decision: RoutingDecision):
        """Append decision to log file."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                entry = {
                    "timestamp": decision.timestamp,
                    "domain": decision.domain,
                    "confidence": round(decision.confidence, 4),
                    "method": decision.method,
                    "specialist": decision.specialist_used,
                    "latency_ms": round(decision.latency_ms, 2),
                }
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def get_analytics(self) -> dict:
        """Get routing analytics summary."""
        total = len(self.decisions)
        if total == 0:
            return {"total_decisions": 0}

        avg_confidence = sum(d.confidence for d in self.decisions) / total
        avg_latency = sum(d.latency_ms for d in self.decisions) / total

        return {
            "total_decisions": total,
            "domain_distribution": dict(self.domain_counts),
            "fallback_rate": self.fallback_count / total,
            "cascade_rate": self.cascade_count / total,
            "avg_confidence": round(avg_confidence, 4),
            "avg_latency_ms": round(avg_latency, 2),
        }

    def print_analytics(self):
        """Display routing analytics."""
        analytics = self.get_analytics()

        cli.rule("Routing Analytics")
        cli.kv_table({
            "Total decisions": str(analytics["total_decisions"]),
            "Avg confidence": f"{analytics.get('avg_confidence', 0):.2%}",
            "Avg latency": f"{analytics.get('avg_latency_ms', 0):.1f}ms",
            "Fallback rate": f"{analytics.get('fallback_rate', 0):.1%}",
            "Cascade rate": f"{analytics.get('cascade_rate', 0):.1%}",
        })

        if analytics.get("domain_distribution"):
            cli.rule("Domain Distribution")
            for domain, count in sorted(analytics["domain_distribution"].items(),
                                       key=lambda x: x[1], reverse=True):
                pct = count / analytics["total_decisions"] * 100
                cli.info(domain, f"{count} ({pct:.0f}%)")
