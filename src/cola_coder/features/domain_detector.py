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
from dataclasses import dataclass, field
from cola_coder.cli import cli

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
