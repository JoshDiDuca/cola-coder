"""Router Training Data Generator: create labeled data for router training.

Uses the domain detection heuristic to auto-label existing code samples,
creating (code_snippet, domain_label) pairs for training the router model.

Output format: JSONL with fields:
  {"code": "...", "domain": "react", "confidence": 0.85, "filename": "App.tsx"}
"""

import json
import numpy as np
from pathlib import Path
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


class RouterDataGenerator:
    """Generate labeled training data for the router model."""

    def __init__(
        self,
        min_confidence: float = 0.3,
        max_samples_per_domain: int = 10000,
        min_code_length: int = 50,
        max_code_length: int = 2000,
    ):
        """
        Args:
            min_confidence: Minimum confidence to include a sample.
            max_samples_per_domain: Cap per domain for balance.
            min_code_length: Minimum code length to include.
            max_code_length: Maximum code length (truncate longer).
        """
        self.min_confidence = min_confidence
        self.max_samples_per_domain = max_samples_per_domain
        self.min_code_length = min_code_length
        self.max_code_length = max_code_length
        self.domain_counts: dict[str, int] = {}
        self.total_processed = 0
        self.total_kept = 0

    def label_code(self, code: str, filename: str = "") -> dict | None:
        """Label a single code sample.

        Args:
            code: Source code string.
            filename: Optional filename.

        Returns:
            Dict with code, domain, confidence, or None if rejected.
        """
        # Length filter
        if len(code) < self.min_code_length:
            return None

        # Truncate if too long
        if len(code) > self.max_code_length:
            code = code[:self.max_code_length]

        # Classify
        from cola_coder.features.domain_detector import detect_domain
        scores = detect_domain(code, filename)

        if not scores or scores[0].confidence < self.min_confidence:
            return None

        domain = scores[0].domain

        # Check domain cap
        current = self.domain_counts.get(domain, 0)
        if current >= self.max_samples_per_domain:
            return None

        self.domain_counts[domain] = current + 1

        return {
            "code": code,
            "domain": domain,
            "confidence": round(scores[0].confidence, 4),
            "filename": filename,
        }

    def generate_from_files(
        self,
        source_dir: str,
        output_path: str = "data/router_training_data.jsonl",
        extensions: tuple[str, ...] = (".ts", ".tsx", ".js", ".jsx"),
    ) -> str:
        """Generate labeled data from a directory of source files.

        Args:
            source_dir: Directory containing source code files.
            output_path: Path to output JSONL file.
            extensions: File extensions to process.

        Returns:
            Path to the output file.
        """
        source_path = Path(source_dir)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cli.header("Cola-Coder", "Router Training Data Generator")
        cli.info("Source directory", str(source_path))
        cli.info("Output", str(output_file))

        self.domain_counts = {}
        self.total_processed = 0
        self.total_kept = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for ext in extensions:
                for filepath in source_path.rglob(f"*{ext}"):
                    try:
                        code = filepath.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue

                    self.total_processed += 1

                    result = self.label_code(code, filepath.name)
                    if result:
                        f.write(json.dumps(result) + "\n")
                        self.total_kept += 1

                    if self.total_processed % 100 == 0:
                        cli.substep(f"Processed {self.total_processed}, kept {self.total_kept}")

        self._print_summary()
        return str(output_file)

    def generate_from_npy(
        self,
        data_path: str,
        tokenizer_path: str = "tokenizer.json",
        output_path: str = "data/router_training_data.jsonl",
        max_samples: int = 50000,
    ) -> str:
        """Generate labeled data from existing .npy training data.

        Decodes tokenized chunks and classifies them.

        Args:
            data_path: Path to .npy data file.
            tokenizer_path: Path to tokenizer.
            output_path: Path to output JSONL file.
            max_samples: Maximum samples to process.

        Returns:
            Path to output file.
        """
        cli.header("Cola-Coder", "Router Data from Training Data")

        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
        tokenizer = CodeTokenizer(tokenizer_path)

        data = np.load(data_path, mmap_mode="r")
        num_chunks = min(data.shape[0], max_samples)

        cli.info("Chunks to process", f"{num_chunks:,}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.domain_counts = {}
        self.total_processed = 0
        self.total_kept = 0

        # Random sample if dataset is larger than max_samples
        if data.shape[0] > max_samples:
            indices = np.random.choice(data.shape[0], max_samples, replace=False)
        else:
            indices = range(num_chunks)

        with open(output_file, "w", encoding="utf-8") as f:
            for idx in indices:
                tokens = data[idx].tolist()
                try:
                    code = tokenizer.decode(tokens)
                except Exception:
                    continue

                self.total_processed += 1

                result = self.label_code(code)
                if result:
                    # Don't store full code for tokenized data, store token indices
                    result["token_indices"] = list(range(len(tokens)))[:256]  # First 256 for router
                    f.write(json.dumps(result) + "\n")
                    self.total_kept += 1

                if self.total_processed % 500 == 0:
                    cli.substep(f"Processed {self.total_processed:,}, kept {self.total_kept:,}")

        self._print_summary()
        return str(output_file)

    def generate_synthetic(
        self,
        output_path: str = "data/router_training_data_synthetic.jsonl",
        samples_per_domain: int = 100,
    ) -> str:
        """Generate synthetic labeled examples for each domain.

        Creates simple template-based examples to bootstrap router training
        when no real labeled data is available.
        """
        templates = {
            "react": [
                "import React from 'react';\n\nexport function {name}() {{\n  const [state, setState] = useState(0);\n  return <div>{name}</div>;\n}}",
                "import {{ useState, useEffect }} from 'react';\n\nconst {name} = () => {{\n  useEffect(() => {{}}, []);\n  return <>{name}</>;\n}};",
            ],
            "nextjs": [
                "import {{ GetServerSideProps }} from 'next';\n\nexport const getServerSideProps: GetServerSideProps = async () => {{\n  return {{ props: {{}} }};\n}};",
                "import {{ useRouter }} from 'next/router';\n\nexport default function {name}() {{\n  const router = useRouter();\n  return <div>{name}</div>;\n}}",
            ],
            "graphql": [
                "import {{ gql }} from 'graphql-tag';\n\nconst {name}_QUERY = gql`\n  query {name} {{\n    items {{ id name }}\n  }}\n`;",
                "type Query {{\n  {name}(id: ID!): {name}\n  all{name}s: [{name}!]!\n}}",
            ],
            "prisma": [
                "import {{ PrismaClient }} from '@prisma/client';\n\nconst prisma = new PrismaClient();\nconst {name}s = await prisma.{name}.findMany();",
            ],
            "zod": [
                "import {{ z }} from 'zod';\n\nconst {name}Schema = z.object({{\n  id: z.string().uuid(),\n  name: z.string().min(1),\n  email: z.string().email(),\n}});\n\ntype {name} = z.infer<typeof {name}Schema>;",
            ],
            "testing": [
                "import {{ describe, it, expect }} from 'vitest';\n\ndescribe('{name}', () => {{\n  it('should work', () => {{\n    expect(true).toBe(true);\n  }});\n}});",
            ],
            "general": [
                "function {name}(x: number): number {{\n  return x * 2;\n}}",
                "const {name}: string[] = ['a', 'b', 'c'];\nconsole.log({name}.length);",
            ],
        }

        names = ["User", "Product", "Order", "Item", "Config", "Setting", "Auth", "Api", "Data", "Service"]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for domain, domain_templates in templates.items():
                for i in range(samples_per_domain):
                    template = domain_templates[i % len(domain_templates)]
                    name = names[i % len(names)]
                    code = template.replace("{name}", name)

                    entry = {
                        "code": code,
                        "domain": domain,
                        "confidence": 1.0,  # Synthetic data has perfect labels
                        "synthetic": True,
                    }
                    f.write(json.dumps(entry) + "\n")
                    count += 1

        cli.success(f"Generated {count} synthetic samples to {output_file}")
        return str(output_file)

    def _print_summary(self):
        """Print generation summary."""
        cli.rule("Generation Summary")
        cli.kv_table({
            "Total processed": f"{self.total_processed:,}",
            "Total kept": f"{self.total_kept:,}",
            "Keep rate": f"{self.total_kept / max(self.total_processed, 1):.1%}",
        })

        if self.domain_counts:
            cli.rule("Domain Distribution")
            for domain, count in sorted(self.domain_counts.items(),
                                       key=lambda x: x[1], reverse=True):
                pct = count / max(self.total_kept, 1) * 100
                cli.info(domain, f"{count:,} ({pct:.0f}%)")
