"""Specialist Registry: manage domain-specific model checkpoints.

A registry that maps domains to specialist model checkpoints.
Supports loading, listing, and managing specialists.

Registry file format (configs/specialists.yaml):
    specialists:
      react:
        checkpoint: checkpoints/react/latest
        config: configs/tiny.yaml
        keywords: [react, jsx, component, hook]
        confidence_threshold: 0.6
      nextjs:
        checkpoint: checkpoints/nextjs/latest
        config: configs/tiny.yaml
        keywords: [next, ssr, ssg, getServerSideProps]
        confidence_threshold: 0.7
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class SpecialistEntry:
    """A registered specialist model."""
    domain: str
    checkpoint_path: str
    config_path: str
    keywords: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.6
    description: str = ""
    is_loaded: bool = False
    model: object = None  # Will hold loaded model reference


class SpecialistRegistry:
    """Registry for domain-specific specialist models."""

    def __init__(self, registry_path: str = "configs/specialists.yaml"):
        """
        Args:
            registry_path: Path to the YAML registry file.
        """
        self.registry_path = Path(registry_path)
        self.specialists: dict[str, SpecialistEntry] = {}
        self._loaded_domain: str | None = None  # Currently loaded specialist

        if self.registry_path.exists():
            self.load_registry()

    def load_registry(self):
        """Load specialist definitions from YAML file."""
        try:
            with open(self.registry_path) as f:
                data = yaml.safe_load(f) or {}

            specs = data.get("specialists", {})
            for domain, info in specs.items():
                self.specialists[domain] = SpecialistEntry(
                    domain=domain,
                    checkpoint_path=info.get("checkpoint", ""),
                    config_path=info.get("config", ""),
                    keywords=info.get("keywords", []),
                    confidence_threshold=info.get("confidence_threshold", 0.6),
                    description=info.get("description", ""),
                )
        except Exception as e:
            cli.warn(f"Failed to load specialist registry: {e}")

    def save_registry(self):
        """Save current registry to YAML file."""
        data = {"specialists": {}}
        for domain, spec in self.specialists.items():
            data["specialists"][domain] = {
                "checkpoint": spec.checkpoint_path,
                "config": spec.config_path,
                "keywords": spec.keywords,
                "confidence_threshold": spec.confidence_threshold,
                "description": spec.description,
            }

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def register(self, domain: str, checkpoint_path: str, config_path: str = "",
                 keywords: list[str] | None = None, confidence_threshold: float = 0.6,
                 description: str = "") -> SpecialistEntry:
        """Register a new specialist.

        Args:
            domain: Domain name (e.g., "react")
            checkpoint_path: Path to model checkpoint
            config_path: Path to model config YAML
            keywords: Domain keywords for routing
            confidence_threshold: Minimum confidence to use this specialist
            description: Human-readable description

        Returns:
            The registered SpecialistEntry.
        """
        entry = SpecialistEntry(
            domain=domain,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            keywords=keywords or [],
            confidence_threshold=confidence_threshold,
            description=description,
        )
        self.specialists[domain] = entry
        self.save_registry()
        cli.success(f"Registered specialist: {domain}")
        return entry

    def unregister(self, domain: str):
        """Remove a specialist from the registry."""
        if domain in self.specialists:
            del self.specialists[domain]
            self.save_registry()
            cli.success(f"Unregistered specialist: {domain}")
        else:
            cli.warn(f"Specialist not found: {domain}")

    def get_specialist(self, domain: str) -> SpecialistEntry | None:
        """Get a specialist by domain name."""
        return self.specialists.get(domain)

    def list_specialists(self) -> list[SpecialistEntry]:
        """List all registered specialists."""
        return list(self.specialists.values())

    def get_available_domains(self) -> list[str]:
        """Get list of domains with available checkpoints."""
        available = []
        for domain, spec in self.specialists.items():
            if Path(spec.checkpoint_path).exists():
                available.append(domain)
        return available

    def print_registry(self):
        """Display the registry as a Rich table."""
        if not self.specialists:
            cli.dim("No specialists registered.")
            cli.dim(f"Register specialists in {self.registry_path}")
            return

        items = {}
        for domain, spec in self.specialists.items():
            ckpt_exists = Path(spec.checkpoint_path).exists() if spec.checkpoint_path else False
            status = "[green]available[/green]" if ckpt_exists else "[red]missing[/red]"
            loaded = " [cyan](loaded)[/cyan]" if spec.is_loaded else ""
            items[domain] = f"{status}{loaded} | {spec.checkpoint_path} | threshold: {spec.confidence_threshold}"

        cli.kv_table(items, title="Specialist Registry")

    def find_by_keywords(self, text: str) -> list[tuple[str, int]]:
        """Find specialists whose keywords match the given text.

        Returns:
            List of (domain, match_count) tuples sorted by match count.
        """
        matches = []
        text_lower = text.lower()
        for domain, spec in self.specialists.items():
            count = sum(1 for kw in spec.keywords if kw.lower() in text_lower)
            if count > 0:
                matches.append((domain, count))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
