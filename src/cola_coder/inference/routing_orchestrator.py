"""Routing orchestrator for domain-aware code generation.

Coordinates semantic routing, specialist loading, memory retrieval,
context assembly, generation, and quality checking.

Flow:
1. Assemble context (memory + repo)
2. Route to domain specialist via SemanticRouter
3. Load specialist model (via HotSwapManager)
4. Generate with specialist (or general fallback)
5. Quality check output
6. Cascade to general if quality poor
7. Update memory from interaction
"""

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    """Record of a routing decision for analytics."""

    domain: str
    confidence: float
    method: str  # "semantic", "heuristic", "fallback"
    specialist_used: bool
    cascade_triggered: bool = False
    latency_ms: float = 0.0
    quality_score: float = 0.0


class RoutingOrchestrator:
    """Orchestrates routing, specialist loading, and generation.

    This is the main integration point that connects:
    - SemanticRouter (domain classification)
    - HotSwapManager (specialist model loading)
    - MemoryManager (project context)
    - CodeGenerator (generation)
    - Quality checking (cascade on poor output)
    """

    def __init__(
        self,
        base_generator,
        semantic_router=None,
        hot_swap_manager=None,
        memory_manager=None,
        confidence_threshold: float = 0.5,
        enable_cascade: bool = True,
        enable_memory: bool = True,
    ):
        """
        Args:
            base_generator: CodeGenerator instance (general model)
            semantic_router: SemanticRouter instance (optional)
            hot_swap_manager: HotSwapManager instance (optional)
            memory_manager: MemoryManager instance (optional)
            confidence_threshold: Min confidence to use specialist
            enable_cascade: Re-route on poor quality output
            enable_memory: Use memory context in prompts
        """
        self.base_generator = base_generator
        self.semantic_router = semantic_router
        self.hot_swap_manager = hot_swap_manager
        self.memory_manager = memory_manager
        self.confidence_threshold = confidence_threshold
        self.enable_cascade = enable_cascade
        self.enable_memory = enable_memory

        # Analytics
        self._decisions: list[RoutingDecision] = []
        self._total_requests = 0

    def generate(
        self,
        prompt: str,
        file_path: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs,
    ) -> tuple[str, RoutingDecision]:
        """Generate code with automatic routing.

        Args:
            prompt: User's prompt
            file_path: Current file path (for context)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold

        Returns:
            (generated_text, routing_decision) tuple
        """
        import time

        start = time.perf_counter()
        self._total_requests += 1

        # 1. Assemble context
        full_prompt = self._build_context(prompt, file_path)

        # 2. Route to domain
        domain, confidence, method = self._route(prompt)

        # 3. Generate with appropriate model
        generator = self._get_generator(domain, confidence)

        output = generator.generate(
            full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        # 4. Quality check + cascade
        cascade_triggered = False
        quality = self._check_quality(output)

        if self.enable_cascade and quality < 0.3 and generator != self.base_generator:
            # Cascade: try general model
            output = self.base_generator.generate(
                full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )
            cascade_triggered = True
            quality = self._check_quality(output)

        # 5. Record decision
        elapsed_ms = (time.perf_counter() - start) * 1000
        decision = RoutingDecision(
            domain=domain,
            confidence=confidence,
            method=method,
            specialist_used=(generator != self.base_generator),
            cascade_triggered=cascade_triggered,
            latency_ms=elapsed_ms,
            quality_score=quality,
        )
        self._decisions.append(decision)

        # 6. Update memory
        if self.enable_memory and self.memory_manager:
            try:
                self.memory_manager.update_from_interaction(prompt, output, domain)
            except Exception:
                pass  # Don't fail generation on memory errors

        return output, decision

    def _build_context(self, prompt: str, file_path: str = "") -> str:
        """Assemble full context: memory + repo + prompt."""
        parts = []

        if self.enable_memory and self.memory_manager:
            try:
                memories = self.memory_manager.get_relevant_memories(
                    query=prompt, file_path=file_path
                )
                if memories:
                    parts.append(memories)
            except Exception:
                pass

        parts.append(prompt)
        return "\n\n".join(parts)

    def _route(self, prompt: str) -> tuple[str, float, str]:
        """Route prompt to a domain.

        Returns:
            (domain, confidence, method) tuple
        """
        if self.semantic_router is None:
            return "general", 1.0, "fallback"

        try:
            memory_ctx = ""
            if self.enable_memory and self.memory_manager:
                memory_ctx = self.memory_manager.get_project_context()

            domain, confidence = self.semantic_router.route_with_context(
                prompt, memory_context=memory_ctx
            )
            return domain, confidence, "semantic"
        except Exception:
            return "general", 1.0, "fallback"

    def _get_generator(self, domain: str, confidence: float):
        """Get the appropriate generator for a domain."""
        if confidence < self.confidence_threshold:
            return self.base_generator

        if domain == "general":
            return self.base_generator

        if self.hot_swap_manager is None:
            return self.base_generator

        # Try to load specialist
        try:
            specialist = self.hot_swap_manager.get(domain)
            if specialist is not None:
                # Return a generator wrapping the specialist model
                # For now, return base generator (specialist integration TBD)
                return self.base_generator
            return self.base_generator
        except Exception:
            return self.base_generator

    def _check_quality(self, output: str) -> float:
        """Quick quality check on generated output.

        Returns:
            Quality score 0-1 (higher = better)
        """
        if not output or len(output.strip()) < 5:
            return 0.0

        score = 0.5  # base

        # Length check
        if len(output) > 20:
            score += 0.1

        # Repetition check
        lines = output.split("\n")
        if len(lines) > 3:
            unique = len(set(lines))
            if unique / len(lines) > 0.5:
                score += 0.2

        # Bracket balance
        opens = output.count("{") + output.count("(") + output.count("[")
        closes = output.count("}") + output.count(")") + output.count("]")
        if opens > 0 and abs(opens - closes) <= 2:
            score += 0.2

        return min(score, 1.0)

    def get_analytics(self) -> dict:
        """Get routing analytics summary."""
        if not self._decisions:
            return {
                "total_requests": self._total_requests,
                "decisions": 0,
            }

        domains: dict[str, int] = {}
        methods: dict[str, int] = {}
        total_cascade = 0
        total_confidence = 0.0

        for d in self._decisions:
            domains[d.domain] = domains.get(d.domain, 0) + 1
            methods[d.method] = methods.get(d.method, 0) + 1
            if d.cascade_triggered:
                total_cascade += 1
            total_confidence += d.confidence

        return {
            "total_requests": self._total_requests,
            "decisions": len(self._decisions),
            "domain_distribution": domains,
            "method_distribution": methods,
            "cascade_rate": total_cascade / max(len(self._decisions), 1),
            "avg_confidence": total_confidence / max(len(self._decisions), 1),
            "avg_latency_ms": sum(d.latency_ms for d in self._decisions) / max(len(self._decisions), 1),
        }

    def reset_analytics(self) -> None:
        """Clear analytics history."""
        self._decisions.clear()
        self._total_requests = 0
