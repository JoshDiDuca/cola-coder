"""Domain Detection Heuristic: classify code by framework/domain.

Uses import analysis and keyword matching to classify TypeScript/JavaScript
code into domains: React, Next.js, GraphQL, Prisma, Zod, Testing, General TS.

This is a fast heuristic baseline - no ML required. Results can be used as
training data for the learned router model.

Domains:
- react: React components, hooks, JSX
- nextjs: Next.js pages, API routes, SSR/SSG
- graphql: GraphQL schemas, resolvers, queries
- prisma: Prisma ORM, database models
- zod: Zod schemas, validation
- testing: Test files (jest, vitest, mocha)
- general: General TypeScript/JavaScript
"""

import re
from dataclasses import dataclass

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED

# Domain definitions with detection patterns
DOMAINS = {
    "react": {
        "imports": [
            r"from\s+['\"]react['\"]",
            r"import\s+React",
            r"from\s+['\"]react-dom['\"]",
            r"from\s+['\"]@radix-ui",
            r"from\s+['\"]@headlessui",
        ],
        "keywords": [
            r"useState\s*\(",
            r"useEffect\s*\(",
            r"useRef\s*\(",
            r"useMemo\s*\(",
            r"useCallback\s*\(",
            r"<[A-Z][a-zA-Z]*",  # JSX components
            r"React\.FC",
            r"React\.Component",
            r"className=",
            r"onClick=",
            r"jsx|tsx",
        ],
        "weight": 1.0,
    },
    "nextjs": {
        "imports": [
            r"from\s+['\"]next/",
            r"from\s+['\"]next['\"]",
            r"import\s+.*\s+from\s+['\"]next/",
        ],
        "keywords": [
            r"getServerSideProps",
            r"getStaticProps",
            r"getStaticPaths",
            r"NextPage",
            r"NextApiRequest",
            r"NextApiResponse",
            r"useRouter\s*\(",
            r"next\.config",
            r"middleware\.ts",
            r"app/.*/(page|layout|loading|error)\.(tsx?|jsx?)",
        ],
        "weight": 1.2,  # Slightly higher weight because Next.js is more specific
    },
    "graphql": {
        "imports": [
            r"from\s+['\"]graphql['\"]",
            r"from\s+['\"]@apollo",
            r"from\s+['\"]urql",
            r"from\s+['\"]graphql-tag",
            r"from\s+['\"]@graphql-codegen",
        ],
        "keywords": [
            r"gql\s*`",
            r"type\s+Query\s*\{",
            r"type\s+Mutation\s*\{",
            r"type\s+Subscription\s*\{",
            r"@Resolver",
            r"@Query\s*\(",
            r"@Mutation\s*\(",
            r"GraphQLSchema",
            r"GraphQLObjectType",
            r"useQuery\s*\(",
            r"useMutation\s*\(",
        ],
        "weight": 1.3,
    },
    "prisma": {
        "imports": [
            r"from\s+['\"]@prisma/client['\"]",
            r"from\s+['\"]prisma['\"]",
        ],
        "keywords": [
            r"PrismaClient",
            r"prisma\.\w+\.(findMany|findUnique|create|update|delete)",
            r"model\s+\w+\s*\{",
            r"@@map",
            r"@@index",
            r"datasource\s+db",
            r"generator\s+client",
        ],
        "weight": 1.3,
    },
    "zod": {
        "imports": [
            r"from\s+['\"]zod['\"]",
            r"import\s+.*z\s+from\s+['\"]zod['\"]",
            r"import\s+\{\s*z\s*\}\s+from\s+['\"]zod['\"]",
        ],
        "keywords": [
            r"z\.object\s*\(",
            r"z\.string\s*\(",
            r"z\.number\s*\(",
            r"z\.array\s*\(",
            r"z\.enum\s*\(",
            r"z\.union\s*\(",
            r"z\.infer\s*<",
            r"\.parse\s*\(",
            r"\.safeParse\s*\(",
        ],
        "weight": 1.2,
    },
    "testing": {
        "imports": [
            r"from\s+['\"]jest['\"]",
            r"from\s+['\"]vitest['\"]",
            r"from\s+['\"]@testing-library",
            r"from\s+['\"]mocha['\"]",
            r"from\s+['\"]chai['\"]",
            r"from\s+['\"]supertest['\"]",
        ],
        "keywords": [
            r"describe\s*\(",
            r"it\s*\(",
            r"\btest\s*\(",
            r"expect\s*\(",
            r"beforeEach\s*\(",
            r"afterEach\s*\(",
            r"beforeAll\s*\(",
            r"afterAll\s*\(",
            r"jest\.mock\s*\(",
            r"vi\.mock\s*\(",
            r"\.test\.(ts|tsx|js|jsx)$",
            r"\.spec\.(ts|tsx|js|jsx)$",
        ],
        "weight": 1.0,
    },
}


@dataclass
class DomainScore:
    """Score for a domain detection."""
    domain: str
    import_matches: int
    keyword_matches: int
    raw_score: float
    confidence: float  # 0-1, normalized


def detect_domain(code: str, filename: str = "") -> list[DomainScore]:
    """Detect the domain of a code snippet.

    Args:
        code: Source code string.
        filename: Optional filename for additional context.

    Returns:
        List of DomainScore sorted by confidence (highest first).
    """
    scores = []

    for domain, patterns in DOMAINS.items():
        import_matches = 0
        keyword_matches = 0

        # Check imports (higher weight)
        for pattern in patterns["imports"]:
            try:
                import_matches += len(re.findall(pattern, code, re.IGNORECASE))
            except re.error:
                pass

        # Check keywords
        for pattern in patterns["keywords"]:
            try:
                keyword_matches += len(re.findall(pattern, code))
            except re.error:
                pass

        # Check filename patterns
        if filename:
            for pattern in patterns.get("keywords", []):
                try:
                    if re.search(pattern, filename):
                        keyword_matches += 2
                except re.error:
                    pass

        # Weighted score: imports count 3x, keywords 1x
        raw_score = (import_matches * 3 + keyword_matches) * patterns["weight"]

        scores.append(DomainScore(
            domain=domain,
            import_matches=import_matches,
            keyword_matches=keyword_matches,
            raw_score=raw_score,
            confidence=0.0,  # Set after normalization
        ))

    # Normalize confidence scores
    total = sum(s.raw_score for s in scores)
    if total > 0:
        for s in scores:
            s.confidence = s.raw_score / total
    else:
        # No matches - everything is "general"
        for s in scores:
            s.confidence = 0.0

    # Sort by confidence
    scores.sort(key=lambda s: s.confidence, reverse=True)

    # If no clear winner, label as general
    if scores and scores[0].confidence < 0.15:
        # Add general with remaining confidence
        general_conf = 1.0 - sum(s.confidence for s in scores)
        scores.append(DomainScore(
            domain="general",
            import_matches=0,
            keyword_matches=0,
            raw_score=0,
            confidence=max(general_conf, 0.5),
        ))
        scores.sort(key=lambda s: s.confidence, reverse=True)

    return scores


def classify(code: str, filename: str = "") -> str:
    """Quick classification: return the top domain name.

    Args:
        code: Source code string.
        filename: Optional filename.

    Returns:
        Domain name string (e.g., "react", "testing", "general")
    """
    scores = detect_domain(code, filename)
    if scores and scores[0].confidence > 0.1:
        return scores[0].domain
    return "general"


@dataclass
class RouteDecision:
    """Margin-aware routing decision for the specialist cascade (MODEL-053).

    The confidence router dispatches to a 50M domain specialist or falls back to
    the general model. A WRONG specialist is worse than the general model, so we
    abstain to "general" when the top domain is not *decisively* ahead — using both
    its absolute confidence AND its MARGIN over the runner-up (selective-prediction
    / cascade routing, arXiv:2502.09054, arXiv:2605.18796).
    """

    domain: str  # chosen specialist domain, or "general" on abstention
    confidence: float  # top-1 normalized confidence
    margin: float  # top-1 minus top-2 confidence (the selective-prediction signal)
    abstained: bool  # True iff we fell back to general for low confidence/margin
    reason: str  # "ok" | "low_confidence" | "low_margin" | "no_signal"


def detection_margin(scores: list[DomainScore]) -> float:
    """Top-1 minus top-2 confidence — the margin used for selective routing.

    Returns the lone top-1 confidence when only one domain scored, and 0.0 for an
    empty list. A small margin means two domains are nearly tied (ambiguous input).
    """
    ranked = sorted(scores, key=lambda s: s.confidence, reverse=True)
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return ranked[0].confidence
    return ranked[0].confidence - ranked[1].confidence


def route_with_abstention(
    code: str,
    filename: str = "",
    min_confidence: float = 0.4,
    min_margin: float = 0.15,
) -> RouteDecision:
    """Route a snippet to a specialist domain, abstaining to "general" when unsure.

    Dispatches to the top domain only when it clears BOTH ``min_confidence`` and a
    ``min_margin`` lead over the runner-up; otherwise it abstains to the general
    model. This is purely additive — it does not change :func:`detect_domain` or
    :func:`classify` (which the router-training-data path uses). MAIN-SAFE (regex).

    Args:
        code: Source code to route.
        filename: Optional filename for extra context.
        min_confidence: Minimum top-1 confidence to trust a specialist (0–1).
        min_margin: Minimum top-1 − top-2 confidence gap to trust a specialist.

    Returns:
        A :class:`RouteDecision`. ``abstained`` is True when it fell back to general
        because the signal was weak (``reason`` explains which guard tripped).
    """
    scores = detect_domain(code, filename)
    if not scores:
        return RouteDecision("general", 0.0, 0.0, True, "no_signal")

    ranked = sorted(scores, key=lambda s: s.confidence, reverse=True)
    top = ranked[0]
    margin = detection_margin(ranked)

    # The heuristic already emits "general" as the top pick for weak input — that
    # is a legitimate (non-abstaining) classification, not a fallback.
    if top.domain == "general":
        return RouteDecision("general", top.confidence, margin, False, "ok")
    if top.confidence < min_confidence:
        return RouteDecision("general", top.confidence, margin, True, "low_confidence")
    if margin < min_margin:
        return RouteDecision("general", top.confidence, margin, True, "low_margin")
    return RouteDecision(top.domain, top.confidence, margin, False, "ok")


@dataclass
class CoveragePoint:
    """One point on the router's risk–coverage curve at a given confidence gate."""
    min_confidence: float
    coverage: float            # fraction of samples routed to a specialist (not abstained/general)
    specialist_accuracy: float # of the covered samples, fraction routed to the CORRECT domain
    n_covered: int             # number of samples routed to a specialist
    n_total: int


def risk_coverage_curve(
    labeled_samples: list[tuple[str, str]],
    confidence_grid: list[float] | None = None,
    min_margin: float = 0.15,
) -> list[CoveragePoint]:
    """Sweep the confidence gate over a labelled corpus → selective-prediction risk–coverage curve.

    Selective prediction trades coverage (how often the router commits to a specialist)
    against risk (how often that commitment is wrong). Raising ``min_confidence`` makes
    the gate stricter: fewer samples are routed (lower coverage) but the ones that are
    should be more reliable (higher specialist accuracy). Plotting accuracy vs. coverage
    across the grid lets the operating threshold be chosen from data instead of a
    hand-picked constant (risk–coverage curve, El-Yaniv & Wiener 2010, arXiv:1901.09192).

    Args:
        labeled_samples: ``(code, true_domain)`` pairs to evaluate.
        confidence_grid: Confidence gates to sweep; defaults to ``0.0``…``0.9`` in 0.1 steps.
        min_margin: Top-1 − top-2 margin floor held fixed across the sweep.

    Returns:
        One :class:`CoveragePoint` per grid value, in grid order. Empty input → ``[]``.
    """
    if not labeled_samples:
        return []

    if confidence_grid is None:
        confidence_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    n_total = len(labeled_samples)
    curve: list[CoveragePoint] = []

    for c in confidence_grid:
        n_covered = 0
        n_correct = 0
        for code, true_domain in labeled_samples:
            decision = route_with_abstention(
                code, min_confidence=c, min_margin=min_margin
            )
            # "Covered" = the router committed to a specialist (not abstained, not general).
            if decision.abstained or decision.domain == "general":
                continue
            n_covered += 1
            if decision.domain == true_domain:
                n_correct += 1

        coverage = n_covered / n_total if n_total > 0 else 0.0
        specialist_accuracy = n_correct / n_covered if n_covered > 0 else 0.0
        curve.append(CoveragePoint(
            min_confidence=c,
            coverage=coverage,
            specialist_accuracy=specialist_accuracy,
            n_covered=n_covered,
            n_total=n_total,
        ))

    return curve


def best_operating_point(
    curve: list[CoveragePoint],
    min_specialist_accuracy: float = 0.8,
) -> CoveragePoint | None:
    """The MAX-coverage point whose specialist_accuracy >= the floor, or None if none qualifies.

    Picks the most permissive gate (highest coverage) that still meets the accuracy
    floor — the natural operating point on a risk–coverage curve.

    Args:
        curve: Points from :func:`risk_coverage_curve`.
        min_specialist_accuracy: Minimum acceptable specialist accuracy (0–1).

    Returns:
        The qualifying :class:`CoveragePoint` with the highest coverage, or ``None``.
    """
    qualifying = [p for p in curve if p.specialist_accuracy >= min_specialist_accuracy]
    if not qualifying:
        return None
    return max(qualifying, key=lambda p: p.coverage)


def batch_classify(code_samples: list[dict]) -> list[dict]:
    """Classify multiple code samples.

    Args:
        code_samples: List of dicts with 'code' and optional 'filename' keys.

    Returns:
        List of dicts with added 'domain' and 'confidence' keys.
    """
    results = []
    for sample in code_samples:
        code = sample.get("code", "")
        filename = sample.get("filename", "")
        scores = detect_domain(code, filename)

        result = dict(sample)
        if scores:
            result["domain"] = scores[0].domain
            result["confidence"] = scores[0].confidence
            result["all_scores"] = {s.domain: round(s.confidence, 3) for s in scores if s.confidence > 0.01}
        else:
            result["domain"] = "general"
            result["confidence"] = 1.0
            result["all_scores"] = {"general": 1.0}

        results.append(result)

    return results
