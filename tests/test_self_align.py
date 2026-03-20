"""Tests for SelfCodeAlign: seed extraction, instruction generation, pipeline.

Run:
    .venv/Scripts/python -m pytest tests/test_self_align.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cola_coder.data.sources.self_align import (
    InstructionExample,
    InstructionGenerator,
    SeedExtractor,
    SelfAlignPipeline,
    SelfAlignSource,
)
from cola_coder.data.pipeline import DataRecord


# ---------------------------------------------------------------------------
# Sample code fixtures
# ---------------------------------------------------------------------------

TS_FUNCTION_CODE = """\
/**
 * Debounce a function call.
 */
export function debounce<T extends (...args: any[]) => void>(
    fn: T,
    delay: number
): (...args: Parameters<T>) => void {
    let timer: ReturnType<typeof setTimeout> | null = null;
    return (...args: Parameters<T>) => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}
"""

TS_CLASS_CODE = """\
class EventEmitter {
    private listeners = new Map<string, Set<Function>>();

    on(event: string, listener: Function): void {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(listener);
    }

    emit(event: string, ...args: any[]): void {
        const handlers = this.listeners.get(event);
        if (handlers) {
            for (const handler of handlers) {
                handler(...args);
            }
        }
    }

    off(event: string, listener: Function): void {
        this.listeners.get(event)?.delete(listener);
    }
}
"""

TS_INTERFACE_CODE = """\
export interface PaginatedResponse<T> {
    data: T[];
    total: number;
    page: number;
    pageSize: number;
    hasNext: boolean;
    hasPrevious: boolean;
}
"""

TS_ARROW_CODE = """\
export const memoize = <T extends (...args: any[]) => any>(fn: T): T => {
    const cache = new Map<string, ReturnType<T>>();
    return ((...args: Parameters<T>): ReturnType<T> => {
        const key = JSON.stringify(args);
        if (cache.has(key)) return cache.get(key)!;
        const result = fn(...args);
        cache.set(key, result);
        return result;
    }) as T;
};
"""

PY_FUNCTION_CODE = """\
def merge_sorted(a: list[int], b: list[int]) -> list[int]:
    \"\"\"Merge two sorted lists into a single sorted list.\"\"\"
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
"""

MULTI_FUNCTION_CODE = """\
function add(a: number, b: number): number {
    return a + b;
}

function subtract(a: number, b: number): number {
    return a - b;
}

function multiply(a: number, b: number): number {
    return a * b;
}
"""

SHORT_CODE = """\
function noop(): void {}
"""


# ---------------------------------------------------------------------------
# SeedExtractor tests
# ---------------------------------------------------------------------------

class TestSeedExtractor:
    """Test seed extraction from various code patterns."""

    def test_extract_ts_function(self):
        extractor = SeedExtractor(min_lines=3)
        seeds = extractor.extract_seeds(TS_FUNCTION_CODE, language="typescript")
        assert len(seeds) >= 1
        assert "debounce" in seeds[0]

    def test_extract_ts_class(self):
        extractor = SeedExtractor(min_lines=3)
        seeds = extractor.extract_seeds(TS_CLASS_CODE, language="typescript")
        assert len(seeds) >= 1
        assert "EventEmitter" in seeds[0]

    def test_extract_ts_interface(self):
        extractor = SeedExtractor(min_lines=3)
        seeds = extractor.extract_seeds(TS_INTERFACE_CODE, language="typescript")
        assert len(seeds) >= 1
        assert "PaginatedResponse" in seeds[0]

    def test_extract_multiple_functions(self):
        extractor = SeedExtractor(min_lines=2)
        seeds = extractor.extract_seeds(MULTI_FUNCTION_CODE, language="typescript")
        assert len(seeds) >= 2  # Should get add, subtract, multiply

    def test_extract_python_function(self):
        extractor = SeedExtractor(min_lines=3)
        seeds = extractor.extract_seeds(PY_FUNCTION_CODE, language="python")
        assert len(seeds) >= 1
        assert "merge_sorted" in seeds[0]

    def test_skip_short_code(self):
        extractor = SeedExtractor(min_lines=3)
        seeds = extractor.extract_seeds(SHORT_CODE, language="typescript")
        assert len(seeds) == 0  # Too short

    def test_max_lines_filter(self):
        extractor = SeedExtractor(min_lines=3, max_lines=5)
        seeds = extractor.extract_seeds(TS_CLASS_CODE, language="typescript")
        # The EventEmitter class is >5 lines, should be filtered out
        assert len(seeds) == 0

    def test_auto_detect_language(self):
        extractor = SeedExtractor(min_lines=3)
        # With language="auto" (falls through to try both)
        seeds = extractor.extract_seeds(PY_FUNCTION_CODE, language="auto")
        assert len(seeds) >= 1


# ---------------------------------------------------------------------------
# InstructionGenerator tests
# ---------------------------------------------------------------------------

class TestInstructionGenerator:
    """Test instruction generation in template mode."""

    def test_template_mode_function(self):
        gen = InstructionGenerator(mode="template")
        example = gen.generate(TS_FUNCTION_CODE.strip(), language="typescript")
        assert example is not None
        assert isinstance(example, InstructionExample)
        assert len(example.instruction) > 10
        assert "debounce" in example.instruction.lower() or "function" in example.instruction.lower()
        assert example.output == TS_FUNCTION_CODE.strip()
        assert 0.0 <= example.quality_score <= 1.0

    def test_template_mode_class(self):
        gen = InstructionGenerator(mode="template")
        example = gen.generate(TS_CLASS_CODE.strip(), language="typescript")
        assert example is not None
        assert "EventEmitter" in example.instruction or "class" in example.instruction.lower()

    def test_template_mode_interface(self):
        gen = InstructionGenerator(mode="template")
        example = gen.generate(TS_INTERFACE_CODE.strip(), language="typescript")
        assert example is not None
        assert "PaginatedResponse" in example.instruction or "interface" in example.instruction.lower()

    def test_template_mode_python(self):
        gen = InstructionGenerator(mode="template")
        example = gen.generate(PY_FUNCTION_CODE.strip(), language="python")
        assert example is not None
        assert len(example.instruction) > 10

    def test_quality_score_range(self):
        gen = InstructionGenerator(mode="template")
        example = gen.generate(TS_FUNCTION_CODE.strip())
        assert example is not None
        assert 0.0 <= example.quality_score <= 1.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            InstructionGenerator(mode="invalid")

    def test_llm_mode_no_key(self):
        """LLM mode without API key should fall back to template."""
        import os
        # Ensure no API keys are set for this test
        old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        old_openai = os.environ.pop("OPENAI_API_KEY", None)
        try:
            gen = InstructionGenerator(mode="llm")
            example = gen.generate(TS_FUNCTION_CODE.strip())
            # Should fall back to template mode and still produce output
            assert example is not None
            assert len(example.instruction) > 10
        finally:
            if old_anthropic is not None:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic
            if old_openai is not None:
                os.environ["OPENAI_API_KEY"] = old_openai

    def test_self_mode_fallback(self):
        """Self mode should fall back to template without a trained model."""
        gen = InstructionGenerator(mode="self")
        example = gen.generate(TS_FUNCTION_CODE.strip())
        assert example is not None
        assert len(example.instruction) > 10


# ---------------------------------------------------------------------------
# InstructionExample tests
# ---------------------------------------------------------------------------

class TestInstructionExample:
    """Test serialization and formatting of InstructionExample."""

    def test_to_dict(self):
        ex = InstructionExample(
            instruction="Write a function",
            input_context="",
            output="function foo() {}",
            seed_code="function foo() {}",
            quality_score=0.8,
        )
        d = ex.to_dict()
        assert d["instruction"] == "Write a function"
        assert d["output"] == "function foo() {}"
        assert d["quality_score"] == 0.8

    def test_from_dict(self):
        d = {
            "instruction": "Write a class",
            "input": "some context",
            "output": "class Foo {}",
            "seed_code": "class Foo {}",
            "quality_score": 0.7,
        }
        ex = InstructionExample.from_dict(d)
        assert ex.instruction == "Write a class"
        assert ex.input_context == "some context"
        assert ex.quality_score == 0.7

    def test_roundtrip_dict(self):
        ex = InstructionExample(
            instruction="Test instruction",
            input_context="context",
            output="output code",
            seed_code="seed",
            quality_score=0.9,
        )
        restored = InstructionExample.from_dict(ex.to_dict())
        assert restored.instruction == ex.instruction
        assert restored.output == ex.output
        assert restored.quality_score == ex.quality_score

    def test_to_training_text(self):
        ex = InstructionExample(
            instruction="Write a function",
            input_context="",
            output="function foo() {}",
            seed_code="",
            quality_score=0.5,
        )
        text = ex.to_training_text()
        assert "### Instruction" in text
        assert "### Response" in text
        assert "Write a function" in text
        assert "function foo() {}" in text

    def test_to_training_text_with_input(self):
        ex = InstructionExample(
            instruction="Refactor this code",
            input_context="old code here",
            output="new code here",
            seed_code="",
            quality_score=0.5,
        )
        text = ex.to_training_text()
        assert "### Input" in text
        assert "old code here" in text

    def test_json_serializable(self):
        ex = InstructionExample(
            instruction="Test",
            input_context="",
            output="code",
            seed_code="seed",
            quality_score=0.5,
        )
        # Should not raise
        json_str = json.dumps(ex.to_dict())
        restored = json.loads(json_str)
        assert restored["instruction"] == "Test"


# ---------------------------------------------------------------------------
# SelfAlignPipeline tests
# ---------------------------------------------------------------------------

class TestSelfAlignPipeline:
    """Test the full pipeline."""

    def test_generate_from_code(self):
        pipeline = SelfAlignPipeline(mode="template", min_quality=0.0)
        examples = pipeline.generate_from_code(TS_FUNCTION_CODE)
        assert len(examples) >= 1
        assert all(isinstance(ex, InstructionExample) for ex in examples)

    def test_generate_from_class_code(self):
        pipeline = SelfAlignPipeline(mode="template", min_quality=0.0)
        examples = pipeline.generate_from_code(TS_CLASS_CODE)
        assert len(examples) >= 1

    def test_quality_filtering(self):
        pipeline_low = SelfAlignPipeline(mode="template", min_quality=0.0)
        pipeline_high = SelfAlignPipeline(mode="template", min_quality=0.99)

        low_examples = pipeline_low.generate_from_code(TS_FUNCTION_CODE)
        high_examples = pipeline_high.generate_from_code(TS_FUNCTION_CODE)

        # High threshold should produce fewer (or equal) results
        assert len(high_examples) <= len(low_examples)

    def test_max_per_file(self):
        pipeline = SelfAlignPipeline(mode="template", min_quality=0.0)
        examples = pipeline.generate_from_code(MULTI_FUNCTION_CODE, max_per_file=1)
        assert len(examples) <= 1

    def test_generate_requires_source(self):
        pipeline = SelfAlignPipeline(source=None, mode="template")
        with pytest.raises(ValueError, match="No data source"):
            pipeline.generate()

    def test_save_and_load_jsonl(self):
        pipeline = SelfAlignPipeline(mode="template", min_quality=0.0)
        examples = pipeline.generate_from_code(TS_FUNCTION_CODE)
        assert len(examples) >= 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_output.jsonl"
            pipeline.save_jsonl(examples, path)

            assert path.exists()
            assert path.stat().st_size > 0

            loaded = SelfAlignPipeline.load_jsonl(path)
            assert len(loaded) == len(examples)
            assert loaded[0].instruction == examples[0].instruction
            assert loaded[0].quality_score == examples[0].quality_score

    def test_python_pipeline(self):
        pipeline = SelfAlignPipeline(mode="template", language="python", min_quality=0.0)
        examples = pipeline.generate_from_code(PY_FUNCTION_CODE)
        assert len(examples) >= 1

    def test_dedup_in_generate(self):
        """Pipeline.generate() should deduplicate by instruction."""
        from cola_coder.data.pipeline import DataSource, DataRecord

        class RepeatSource(DataSource):
            """Yields the same code twice."""
            def name(self) -> str:
                return "repeat"
            def stream(self):
                for _ in range(3):
                    yield DataRecord(content=TS_FUNCTION_CODE, metadata={})

        pipeline = SelfAlignPipeline(
            source=RepeatSource(),
            mode="template",
            min_quality=0.0,
        )
        examples = pipeline.generate(max_examples=100)
        # Despite 3 copies of the same code, instructions should be deduped
        instructions = [ex.instruction.lower().strip() for ex in examples]
        assert len(instructions) == len(set(instructions))


# ---------------------------------------------------------------------------
# SelfAlignSource (DataSource adapter) tests
# ---------------------------------------------------------------------------

class TestSelfAlignSource:
    """Test the DataSource adapter."""

    def test_name(self):
        source = SelfAlignSource(mode="template", max_examples=500)
        assert "self_align" in source.name()
        assert "template" in source.name()

    def test_estimate_size(self):
        source = SelfAlignSource(max_examples=2000)
        assert source.estimate_size() == 2000

    def test_stream_without_source_yields_nothing(self):
        """With no inner source configured, stream yields nothing."""
        source = SelfAlignSource(mode="template")
        records = list(source.stream())
        assert len(records) == 0

    def test_stream_with_local_source(self):
        """With a local source pointing to temp files, stream yields records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a TS file
            ts_file = Path(tmpdir) / "example.ts"
            ts_file.write_text(TS_FUNCTION_CODE, encoding="utf-8")

            source = SelfAlignSource(
                mode="template",
                max_examples=10,
                language="typescript",
                min_quality=0.0,
                source_type="local",
                source_paths=[tmpdir],
            )
            records = list(source.stream())
            assert len(records) >= 1
            assert isinstance(records[0], DataRecord)
            assert "### Instruction" in records[0].content
            assert "### Response" in records[0].content
            assert records[0].metadata["source"] == "self_align"
