"""TypeScript-specific code generation benchmark.

50 problems covering the domains cola-coder targets: basics, types, React,
Next.js, Prisma, Zod, and testing.

Evaluation is purely static (no TypeScript runtime required):
  Tier 1 – structural check: expected function/class name present
  Tier 2 – type annotation check: TypeScript-specific syntax present
  Tier 3 – pattern matching: domain-specific patterns (useState, schema, etc.)
  Tier 4 – optional tsc type-check (only runs when `tsc` is on PATH)

For a TS dev: think of this like a Jest test suite for your own code generator.
Each TSProblem is one test case; TSBenchmark runs them all and reports pass@1.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TSProblem:
    """A single TypeScript coding problem."""

    id: str
    prompt: str  # The code prefix shown to the model
    canonical_solution: str  # Reference correct solution
    test_code: str  # TypeScript test (for documentation / optional runtime)
    category: str  # basics | types | react | nextjs | prisma | zod | testing
    difficulty: str  # easy | medium | hard
    description: str
    required_patterns: list[str] = field(default_factory=list)  # regex patterns
    forbidden_patterns: list[str] = field(default_factory=list)  # must NOT appear


@dataclass
class TSBenchmarkResult:
    """Aggregate results from a full benchmark run."""

    total_problems: int
    solved: int
    pass_rate: float
    by_category: dict[str, float]
    by_difficulty: dict[str, float]
    details: list[dict]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "TYPESCRIPT BENCHMARK RESULTS",
            "=" * 60,
            f"  Overall pass rate: {self.pass_rate:.1%}  ({self.solved}/{self.total_problems})",
            "",
            "  By category:",
        ]
        for cat, rate in sorted(self.by_category.items()):
            lines.append(f"    {cat:<12} {rate:.1%}")
        lines.append("")
        lines.append("  By difficulty:")
        for diff, rate in sorted(self.by_difficulty.items()):
            lines.append(f"    {diff:<12} {rate:.1%}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Problem definitions – 50 problems
# ---------------------------------------------------------------------------

PROBLEMS: list[TSProblem] = [
    # -----------------------------------------------------------------------
    # BASICS (10 problems)
    # -----------------------------------------------------------------------
    TSProblem(
        id="basics_fibonacci",
        category="basics",
        difficulty="easy",
        description="Compute the n-th Fibonacci number",
        prompt=(
            "/**\n"
            " * Returns the n-th Fibonacci number (0-indexed).\n"
            " * fib(0) => 0, fib(1) => 1, fib(10) => 55\n"
            " */\n"
            "function fibonacci(n: number): number {"
        ),
        canonical_solution=(
            "function fibonacci(n: number): number {\n"
            "  if (n <= 1) return n;\n"
            "  let a = 0, b = 1;\n"
            "  for (let i = 2; i <= n; i++) { [a, b] = [b, a + b]; }\n"
            "  return b;\n"
            "}"
        ),
        test_code=(
            "import { fibonacci } from './solution';\n"
            "test('fibonacci', () => {\n"
            "  expect(fibonacci(0)).toBe(0);\n"
            "  expect(fibonacci(1)).toBe(1);\n"
            "  expect(fibonacci(10)).toBe(55);\n"
            "});\n"
        ),
        required_patterns=[r"function fibonacci", r":\s*number"],
    ),
    TSProblem(
        id="basics_is_palindrome",
        category="basics",
        difficulty="easy",
        description="Check whether a string is a palindrome",
        prompt=(
            "/**\n"
            " * Returns true if `s` reads the same forwards and backwards.\n"
            " */\n"
            "function isPalindrome(s: string): boolean {"
        ),
        canonical_solution=(
            "function isPalindrome(s: string): boolean {\n"
            "  const cleaned = s.toLowerCase().replace(/[^a-z0-9]/g, '');\n"
            "  return cleaned === cleaned.split('').reverse().join('');\n"
            "}"
        ),
        test_code=(
            "test('isPalindrome', () => {\n"
            "  expect(isPalindrome('racecar')).toBe(true);\n"
            "  expect(isPalindrome('hello')).toBe(false);\n"
            "});\n"
        ),
        required_patterns=[r"function isPalindrome", r":\s*boolean"],
    ),
    TSProblem(
        id="basics_flatten",
        category="basics",
        difficulty="easy",
        description="Flatten a nested array one level deep",
        prompt=(
            "/**\n"
            " * Flattens an array one level deep.\n"
            " * flatten([[1,2],[3,[4]]]) => [1, 2, 3, [4]]\n"
            " */\n"
            "function flatten<T>(arr: T[][]): T[] {"
        ),
        canonical_solution=(
            "function flatten<T>(arr: T[][]): T[] {\n"
            "  return arr.reduce((acc, val) => acc.concat(val), [] as T[]);\n"
            "}"
        ),
        test_code=(
            "test('flatten', () => {\n"
            "  expect(flatten([[1, 2], [3, 4]])).toEqual([1, 2, 3, 4]);\n"
            "});\n"
        ),
        required_patterns=[r"function flatten", r"<T>"],
    ),
    TSProblem(
        id="basics_debounce",
        category="basics",
        difficulty="medium",
        description="Implement a debounce utility",
        prompt=(
            "/**\n"
            " * Returns a debounced version of `fn` that delays invocation\n"
            " * until `delay` ms have elapsed since the last call.\n"
            " */\n"
            "function debounce<T extends (...args: unknown[]) => void>(\n"
            "  fn: T,\n"
            "  delay: number,\n"
            "): (...args: Parameters<T>) => void {"
        ),
        canonical_solution=(
            "function debounce<T extends (...args: unknown[]) => void>(\n"
            "  fn: T,\n"
            "  delay: number,\n"
            "): (...args: Parameters<T>) => void {\n"
            "  let timer: ReturnType<typeof setTimeout> | null = null;\n"
            "  return (...args: Parameters<T>) => {\n"
            "    if (timer) clearTimeout(timer);\n"
            "    timer = setTimeout(() => fn(...args), delay);\n"
            "  };\n"
            "}"
        ),
        test_code=(
            "test('debounce delays call', () => {\n"
            "  jest.useFakeTimers();\n"
            "  const fn = jest.fn();\n"
            "  const dFn = debounce(fn, 100);\n"
            "  dFn(); dFn(); dFn();\n"
            "  jest.runAllTimers();\n"
            "  expect(fn).toHaveBeenCalledTimes(1);\n"
            "});\n"
        ),
        required_patterns=[r"function debounce", r"setTimeout", r"Parameters<T>"],
    ),
    TSProblem(
        id="basics_chunk",
        category="basics",
        difficulty="easy",
        description="Split an array into chunks of a given size",
        prompt=(
            "/**\n"
            " * Splits `arr` into sub-arrays of length `size`.\n"
            " * chunk([1,2,3,4,5], 2) => [[1,2],[3,4],[5]]\n"
            " */\n"
            "function chunk<T>(arr: T[], size: number): T[][] {"
        ),
        canonical_solution=(
            "function chunk<T>(arr: T[], size: number): T[][] {\n"
            "  const result: T[][] = [];\n"
            "  for (let i = 0; i < arr.length; i += size) {\n"
            "    result.push(arr.slice(i, i + size));\n"
            "  }\n"
            "  return result;\n"
            "}"
        ),
        test_code=(
            "test('chunk', () => {\n"
            "  expect(chunk([1,2,3,4,5], 2)).toEqual([[1,2],[3,4],[5]]);\n"
            "});\n"
        ),
        required_patterns=[r"function chunk", r"<T>", r"T\[\]\[\]"],
    ),
    TSProblem(
        id="basics_unique",
        category="basics",
        difficulty="easy",
        description="Return unique values from an array",
        prompt=(
            "/**\n"
            " * Returns an array of unique values from `arr` (preserving order).\n"
            " */\n"
            "function unique<T>(arr: T[]): T[] {"
        ),
        canonical_solution=(
            "function unique<T>(arr: T[]): T[] {\n"
            "  return [...new Set(arr)];\n"
            "}"
        ),
        test_code=(
            "test('unique', () => {\n"
            "  expect(unique([1, 2, 1, 3, 2])).toEqual([1, 2, 3]);\n"
            "});\n"
        ),
        required_patterns=[r"function unique", r"Set"],
    ),
    TSProblem(
        id="basics_group_by",
        category="basics",
        difficulty="medium",
        description="Group array items by a key-selector function",
        prompt=(
            "/**\n"
            " * Groups `arr` items into a Record keyed by the result of `keyFn`.\n"
            " */\n"
            "function groupBy<T, K extends string | number>(\n"
            "  arr: T[],\n"
            "  keyFn: (item: T) => K,\n"
            "): Record<K, T[]> {"
        ),
        canonical_solution=(
            "function groupBy<T, K extends string | number>(\n"
            "  arr: T[],\n"
            "  keyFn: (item: T) => K,\n"
            "): Record<K, T[]> {\n"
            "  return arr.reduce((acc, item) => {\n"
            "    const key = keyFn(item);\n"
            "    (acc[key] = acc[key] ?? []).push(item);\n"
            "    return acc;\n"
            "  }, {} as Record<K, T[]>);\n"
            "}"
        ),
        test_code=(
            "test('groupBy', () => {\n"
            "  const result = groupBy([1, 2, 3, 4], x => x % 2 === 0 ? 'even' : 'odd');\n"
            "  expect(result.even).toEqual([2, 4]);\n"
            "  expect(result.odd).toEqual([1, 3]);\n"
            "});\n"
        ),
        required_patterns=[r"function groupBy", r"Record<K,\s*T\[\]>|Record<K, T\[\]>"],
    ),
    TSProblem(
        id="basics_sleep",
        category="basics",
        difficulty="easy",
        description="Return a Promise that resolves after `ms` milliseconds",
        prompt=(
            "/**\n"
            " * Returns a Promise that resolves after `ms` milliseconds.\n"
            " */\n"
            "function sleep(ms: number): Promise<void> {"
        ),
        canonical_solution=(
            "function sleep(ms: number): Promise<void> {\n"
            "  return new Promise(resolve => setTimeout(resolve, ms));\n"
            "}"
        ),
        test_code=(
            "test('sleep resolves after delay', async () => {\n"
            "  jest.useFakeTimers();\n"
            "  const p = sleep(100);\n"
            "  jest.runAllTimers();\n"
            "  await p;\n"
            "});\n"
        ),
        required_patterns=[r"function sleep", r"Promise<void>", r"setTimeout"],
    ),
    TSProblem(
        id="basics_clamp",
        category="basics",
        difficulty="easy",
        description="Clamp a number between min and max",
        prompt=(
            "/**\n"
            " * Clamps `value` between `min` and `max` (inclusive).\n"
            " */\n"
            "function clamp(value: number, min: number, max: number): number {"
        ),
        canonical_solution=(
            "function clamp(value: number, min: number, max: number): number {\n"
            "  return Math.min(max, Math.max(min, value));\n"
            "}"
        ),
        test_code=(
            "test('clamp', () => {\n"
            "  expect(clamp(5, 0, 10)).toBe(5);\n"
            "  expect(clamp(-5, 0, 10)).toBe(0);\n"
            "  expect(clamp(15, 0, 10)).toBe(10);\n"
            "});\n"
        ),
        required_patterns=[r"function clamp", r":\s*number"],
    ),
    TSProblem(
        id="basics_pick",
        category="basics",
        difficulty="medium",
        description="Pick specific keys from an object",
        prompt=(
            "/**\n"
            " * Returns a new object with only the specified keys picked from `obj`.\n"
            " */\n"
            "function pick<T extends object, K extends keyof T>(\n"
            "  obj: T,\n"
            "  keys: K[],\n"
            "): Pick<T, K> {"
        ),
        canonical_solution=(
            "function pick<T extends object, K extends keyof T>(\n"
            "  obj: T,\n"
            "  keys: K[],\n"
            "): Pick<T, K> {\n"
            "  return keys.reduce((acc, key) => {\n"
            "    acc[key] = obj[key];\n"
            "    return acc;\n"
            "  }, {} as Pick<T, K>);\n"
            "}"
        ),
        test_code=(
            "test('pick', () => {\n"
            "  expect(pick({ a: 1, b: 2, c: 3 }, ['a', 'c'])).toEqual({ a: 1, c: 3 });\n"
            "});\n"
        ),
        required_patterns=[r"function pick", r"keyof T", r"Pick<T,\s*K>|Pick<T, K>"],
    ),
    # -----------------------------------------------------------------------
    # TYPES (8 problems)
    # -----------------------------------------------------------------------
    TSProblem(
        id="types_generic_identity",
        category="types",
        difficulty="easy",
        description="Generic identity function",
        prompt=(
            "/**\n"
            " * Returns its argument unchanged, preserving the type.\n"
            " */\n"
            "function identity<T>(value: T): T {"
        ),
        canonical_solution="function identity<T>(value: T): T {\n  return value;\n}",
        test_code=(
            "test('identity', () => {\n"
            "  expect(identity(42)).toBe(42);\n"
            "  expect(identity('hello')).toBe('hello');\n"
            "});\n"
        ),
        required_patterns=[r"function identity", r"<T>", r"\(value:\s*T\):\s*T"],
    ),
    TSProblem(
        id="types_partial_update",
        category="types",
        difficulty="medium",
        description="Merge a partial update into an existing object",
        prompt=(
            "/**\n"
            " * Returns a new object that is `base` with `updates` merged in.\n"
            " * Type-safe: updates must be Partial<T>.\n"
            " */\n"
            "function mergeUpdate<T extends object>(base: T, updates: Partial<T>): T {"
        ),
        canonical_solution=(
            "function mergeUpdate<T extends object>(base: T, updates: Partial<T>): T {\n"
            "  return { ...base, ...updates };\n"
            "}"
        ),
        test_code=(
            "test('mergeUpdate', () => {\n"
            "  expect(mergeUpdate({ a: 1, b: 2 }, { b: 99 })).toEqual({ a: 1, b: 99 });\n"
            "});\n"
        ),
        required_patterns=[r"function mergeUpdate", r"Partial<T>", r"<T extends object>"],
    ),
    TSProblem(
        id="types_type_guard_string",
        category="types",
        difficulty="easy",
        description="Type guard to narrow unknown to string",
        prompt=(
            "/**\n"
            " * Returns true if `value` is a string.\n"
            " */\n"
            "function isString(value: unknown): value is string {"
        ),
        canonical_solution=(
            "function isString(value: unknown): value is string {\n"
            "  return typeof value === 'string';\n"
            "}"
        ),
        test_code=(
            "test('isString', () => {\n"
            "  expect(isString('hello')).toBe(true);\n"
            "  expect(isString(42)).toBe(false);\n"
            "});\n"
        ),
        required_patterns=[r"function isString", r"value is string"],
    ),
    TSProblem(
        id="types_discriminated_union",
        category="types",
        difficulty="medium",
        description="Handle a discriminated union with exhaustive switch",
        prompt=(
            "type Shape =\n"
            "  | { kind: 'circle'; radius: number }\n"
            "  | { kind: 'rect'; width: number; height: number };\n"
            "\n"
            "/**\n"
            " * Computes the area of a Shape.\n"
            " */\n"
            "function area(shape: Shape): number {"
        ),
        canonical_solution=(
            "function area(shape: Shape): number {\n"
            "  switch (shape.kind) {\n"
            "    case 'circle': return Math.PI * shape.radius ** 2;\n"
            "    case 'rect':   return shape.width * shape.height;\n"
            "  }\n"
            "}"
        ),
        test_code=(
            "test('area', () => {\n"
            "  expect(area({ kind: 'rect', width: 4, height: 5 })).toBe(20);\n"
            "});\n"
        ),
        required_patterns=[r"function area", r"switch", r"case 'circle'|case \"circle\""],
    ),
    TSProblem(
        id="types_mapped_type_readonly",
        category="types",
        difficulty="medium",
        description="Create a deep-readonly mapped type",
        prompt=(
            "/**\n"
            " * Mapped type that makes every property of T readonly recursively.\n"
            " */\n"
            "type DeepReadonly<T> ="
        ),
        canonical_solution=(
            "type DeepReadonly<T> = {\n"
            "  readonly [K in keyof T]: T[K] extends object ? DeepReadonly<T[K]> : T[K];\n"
            "};"
        ),
        test_code=(
            "type Obj = { a: { b: number } };\n"
            "type ROObj = DeepReadonly<Obj>; // should not allow mutation\n"
        ),
        required_patterns=[r"type DeepReadonly", r"readonly", r"keyof T"],
    ),
    TSProblem(
        id="types_record_builder",
        category="types",
        difficulty="medium",
        description="Build a Record from an array using a key extractor",
        prompt=(
            "/**\n"
            " * Converts an array of objects into a Record keyed by `keyFn`.\n"
            " */\n"
            "function toRecord<T, K extends string>(\n"
            "  arr: T[],\n"
            "  keyFn: (item: T) => K,\n"
            "): Record<K, T> {"
        ),
        canonical_solution=(
            "function toRecord<T, K extends string>(\n"
            "  arr: T[],\n"
            "  keyFn: (item: T) => K,\n"
            "): Record<K, T> {\n"
            "  return arr.reduce((acc, item) => {\n"
            "    acc[keyFn(item)] = item;\n"
            "    return acc;\n"
            "  }, {} as Record<K, T>);\n"
            "}"
        ),
        test_code=(
            "test('toRecord', () => {\n"
            "  const users = [{ id: 'a', name: 'Alice' }, { id: 'b', name: 'Bob' }];\n"
            "  const rec = toRecord(users, u => u.id);\n"
            "  expect(rec['a'].name).toBe('Alice');\n"
            "});\n"
        ),
        required_patterns=[r"function toRecord", r"Record<K,\s*T>|Record<K, T>"],
    ),
    TSProblem(
        id="types_extract_promise",
        category="types",
        difficulty="hard",
        description="Utility type to unwrap a Promise type",
        prompt=(
            "/**\n"
            " * Extracts the resolved type from a Promise.\n"
            " * UnwrapPromise<Promise<number>> => number\n"
            " */\n"
            "type UnwrapPromise<T> ="
        ),
        canonical_solution=(
            "type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;"
        ),
        test_code="type N = UnwrapPromise<Promise<number>>; // should be number\n",
        required_patterns=[r"type UnwrapPromise", r"infer\s+U", r"Promise<infer"],
    ),
    TSProblem(
        id="types_non_nullable",
        category="types",
        difficulty="easy",
        description="Filter null/undefined from a type",
        prompt=(
            "/**\n"
            " * A utility type alias for NonNullable applied to all values of T.\n"
            " * RequiredValues<{ a: string | null; b: number | undefined }>\n"
            " *   => { a: string; b: number }\n"
            " */\n"
            "type RequiredValues<T> ="
        ),
        canonical_solution=(
            "type RequiredValues<T> = { [K in keyof T]: NonNullable<T[K]> };"
        ),
        test_code=(
            "type R = RequiredValues<{ a: string | null; b: number | undefined }>;\n"
            "// R should be { a: string; b: number }\n"
        ),
        required_patterns=[r"type RequiredValues", r"NonNullable", r"keyof T"],
    ),
    # -----------------------------------------------------------------------
    # REACT (8 problems)
    # -----------------------------------------------------------------------
    TSProblem(
        id="react_use_state_counter",
        category="react",
        difficulty="easy",
        description="Counter component with useState",
        prompt=(
            "import React, { useState } from 'react';\n"
            "\n"
            "/**\n"
            " * A simple counter component with increment and decrement buttons.\n"
            " */\n"
            "export function Counter(): React.ReactElement {"
        ),
        canonical_solution=(
            "export function Counter(): React.ReactElement {\n"
            "  const [count, setCount] = useState(0);\n"
            "  return (\n"
            "    <div>\n"
            "      <button onClick={() => setCount(c => c - 1)}>-</button>\n"
            "      <span>{count}</span>\n"
            "      <button onClick={() => setCount(c => c + 1)}>+</button>\n"
            "    </div>\n"
            "  );\n"
            "}"
        ),
        test_code=(
            "test('counter increments', () => {\n"
            "  const { getByText } = render(<Counter />);\n"
            "  fireEvent.click(getByText('+'));\n"
            "  expect(getByText('1')).toBeInTheDocument();\n"
            "});\n"
        ),
        required_patterns=[r"useState", r"setCount|set[A-Z]", r"onClick"],
    ),
    TSProblem(
        id="react_use_effect_cleanup",
        category="react",
        difficulty="medium",
        description="useEffect with event listener and cleanup",
        prompt=(
            "import { useEffect } from 'react';\n"
            "\n"
            "/**\n"
            " * Custom hook that calls `handler` whenever a key is pressed.\n"
            " * Properly cleans up the event listener on unmount.\n"
            " */\n"
            "export function useKeyPress(\n"
            "  key: string,\n"
            "  handler: (event: KeyboardEvent) => void,\n"
            "): void {"
        ),
        canonical_solution=(
            "export function useKeyPress(\n"
            "  key: string,\n"
            "  handler: (event: KeyboardEvent) => void,\n"
            "): void {\n"
            "  useEffect(() => {\n"
            "    const listener = (event: KeyboardEvent) => {\n"
            "      if (event.key === key) handler(event);\n"
            "    };\n"
            "    window.addEventListener('keydown', listener);\n"
            "    return () => window.removeEventListener('keydown', listener);\n"
            "  }, [key, handler]);\n"
            "}"
        ),
        test_code=(
            "test('cleans up listener', () => {\n"
            "  const spy = jest.spyOn(window, 'removeEventListener');\n"
            "  const { unmount } = renderHook(() => useKeyPress('Enter', jest.fn()));\n"
            "  unmount();\n"
            "  expect(spy).toHaveBeenCalledWith('keydown', expect.any(Function));\n"
            "});\n"
        ),
        required_patterns=[r"useEffect", r"addEventListener", r"removeEventListener", r"return\s*\(\s*\)"],
    ),
    TSProblem(
        id="react_custom_hook_fetch",
        category="react",
        difficulty="medium",
        description="useFetch custom hook for data fetching",
        prompt=(
            "import { useState, useEffect } from 'react';\n"
            "\n"
            "interface FetchState<T> {\n"
            "  data: T | null;\n"
            "  loading: boolean;\n"
            "  error: Error | null;\n"
            "}\n"
            "\n"
            "/**\n"
            " * Custom hook that fetches `url` and returns { data, loading, error }.\n"
            " */\n"
            "export function useFetch<T>(url: string): FetchState<T> {"
        ),
        canonical_solution=(
            "export function useFetch<T>(url: string): FetchState<T> {\n"
            "  const [state, setState] = useState<FetchState<T>>({ data: null, loading: true, error: null });\n"
            "  useEffect(() => {\n"
            "    let cancelled = false;\n"
            "    fetch(url)\n"
            "      .then(r => r.json() as Promise<T>)\n"
            "      .then(data => { if (!cancelled) setState({ data, loading: false, error: null }); })\n"
            "      .catch(error => { if (!cancelled) setState({ data: null, loading: false, error }); });\n"
            "    return () => { cancelled = true; };\n"
            "  }, [url]);\n"
            "  return state;\n"
            "}"
        ),
        test_code=(
            "test('returns loading true initially', () => {\n"
            "  const { result } = renderHook(() => useFetch<string>('/api/test'));\n"
            "  expect(result.current.loading).toBe(true);\n"
            "});\n"
        ),
        required_patterns=[r"useFetch", r"useState", r"useEffect", r"fetch\("],
    ),
    TSProblem(
        id="react_props_interface",
        category="react",
        difficulty="easy",
        description="Component with typed props interface",
        prompt=(
            "import React from 'react';\n"
            "\n"
            "interface ButtonProps {\n"
            "  label: string;\n"
            "  onClick: () => void;\n"
            "  disabled?: boolean;\n"
            "  variant?: 'primary' | 'secondary';\n"
            "}\n"
            "\n"
            "/**\n"
            " * A reusable Button component with full TypeScript prop types.\n"
            " */\n"
            "export function Button({ label, onClick, disabled = false, variant = 'primary' }: ButtonProps): React.ReactElement {"
        ),
        canonical_solution=(
            "export function Button({ label, onClick, disabled = false, variant = 'primary' }: ButtonProps): React.ReactElement {\n"
            "  return (\n"
            "    <button\n"
            "      className={`btn btn-${variant}`}\n"
            "      onClick={onClick}\n"
            "      disabled={disabled}\n"
            "    >\n"
            "      {label}\n"
            "    </button>\n"
            "  );\n"
            "}"
        ),
        test_code=(
            "test('renders label', () => {\n"
            "  const { getByText } = render(<Button label='Click me' onClick={jest.fn()} />);\n"
            "  expect(getByText('Click me')).toBeInTheDocument();\n"
            "});\n"
        ),
        required_patterns=[r"ButtonProps", r"onClick", r"disabled"],
    ),
    TSProblem(
        id="react_use_ref",
        category="react",
        difficulty="medium",
        description="Component using useRef for DOM focus",
        prompt=(
            "import React, { useRef, useEffect } from 'react';\n"
            "\n"
            "/**\n"
            " * Input that automatically focuses itself on mount.\n"
            " */\n"
            "export function AutoFocusInput(): React.ReactElement {"
        ),
        canonical_solution=(
            "export function AutoFocusInput(): React.ReactElement {\n"
            "  const inputRef = useRef<HTMLInputElement>(null);\n"
            "  useEffect(() => {\n"
            "    inputRef.current?.focus();\n"
            "  }, []);\n"
            "  return <input ref={inputRef} />;\n"
            "}"
        ),
        test_code=(
            "test('focuses on mount', () => {\n"
            "  const { container } = render(<AutoFocusInput />);\n"
            "  expect(document.activeElement).toBe(container.querySelector('input'));\n"
            "});\n"
        ),
        required_patterns=[r"useRef", r"useEffect", r"\.focus\(\)"],
    ),
    TSProblem(
        id="react_use_reducer",
        category="react",
        difficulty="medium",
        description="useReducer with typed action union",
        prompt=(
            "import React, { useReducer } from 'react';\n"
            "\n"
            "type Action =\n"
            "  | { type: 'increment' }\n"
            "  | { type: 'decrement' }\n"
            "  | { type: 'reset' };\n"
            "\n"
            "/**\n"
            " * Reducer for a counter state.\n"
            " */\n"
            "function counterReducer(state: number, action: Action): number {"
        ),
        canonical_solution=(
            "function counterReducer(state: number, action: Action): number {\n"
            "  switch (action.type) {\n"
            "    case 'increment': return state + 1;\n"
            "    case 'decrement': return state - 1;\n"
            "    case 'reset':     return 0;\n"
            "  }\n"
            "}"
        ),
        test_code=(
            "test('reducer increment', () => {\n"
            "  expect(counterReducer(0, { type: 'increment' })).toBe(1);\n"
            "});\n"
        ),
        required_patterns=[r"counterReducer", r"switch", r"case 'increment'|case \"increment\""],
    ),
    TSProblem(
        id="react_context",
        category="react",
        difficulty="medium",
        description="Create a typed React context with provider",
        prompt=(
            "import React, { createContext, useContext, useState, ReactNode } from 'react';\n"
            "\n"
            "interface ThemeContextValue {\n"
            "  theme: 'light' | 'dark';\n"
            "  toggleTheme: () => void;\n"
            "}\n"
            "\n"
            "const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);\n"
            "\n"
            "/**\n"
            " * Provider component for the theme context.\n"
            " */\n"
            "export function ThemeProvider({ children }: { children: ReactNode }): React.ReactElement {"
        ),
        canonical_solution=(
            "export function ThemeProvider({ children }: { children: ReactNode }): React.ReactElement {\n"
            "  const [theme, setTheme] = useState<'light' | 'dark'>('light');\n"
            "  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light');\n"
            "  return (\n"
            "    <ThemeContext.Provider value={{ theme, toggleTheme }}>\n"
            "      {children}\n"
            "    </ThemeContext.Provider>\n"
            "  );\n"
            "}"
        ),
        test_code=(
            "test('provides theme context', () => {\n"
            "  const { result } = renderHook(() => useContext(ThemeContext), {\n"
            "    wrapper: ThemeProvider,\n"
            "  });\n"
            "  expect(result.current?.theme).toBe('light');\n"
            "});\n"
        ),
        required_patterns=[r"ThemeProvider", r"useState", r"ThemeContext\.Provider|ThemeContext.Provider"],
    ),
    TSProblem(
        id="react_memo_component",
        category="react",
        difficulty="medium",
        description="Memoized component with React.memo",
        prompt=(
            "import React from 'react';\n"
            "\n"
            "interface ListItemProps {\n"
            "  text: string;\n"
            "  selected: boolean;\n"
            "  onSelect: (text: string) => void;\n"
            "}\n"
            "\n"
            "/**\n"
            " * A list item component memoized to avoid unnecessary re-renders.\n"
            " */\n"
            "export const ListItem = React.memo(function ListItem({\n"
            "  text,\n"
            "  selected,\n"
            "  onSelect,\n"
            "}: ListItemProps): React.ReactElement {"
        ),
        canonical_solution=(
            "export const ListItem = React.memo(function ListItem({\n"
            "  text,\n"
            "  selected,\n"
            "  onSelect,\n"
            "}: ListItemProps): React.ReactElement {\n"
            "  return (\n"
            "    <li\n"
            "      className={selected ? 'selected' : ''}\n"
            "      onClick={() => onSelect(text)}\n"
            "    >\n"
            "      {text}\n"
            "    </li>\n"
            "  );\n"
            "});"
        ),
        test_code=(
            "test('calls onSelect on click', () => {\n"
            "  const onSelect = jest.fn();\n"
            "  const { getByText } = render(<ListItem text='foo' selected={false} onSelect={onSelect} />);\n"
            "  fireEvent.click(getByText('foo'));\n"
            "  expect(onSelect).toHaveBeenCalledWith('foo');\n"
            "});\n"
        ),
        required_patterns=[r"React\.memo|React.memo", r"onClick", r"ListItemProps"],
    ),
    # -----------------------------------------------------------------------
    # NEXT.JS (6 problems)
    # -----------------------------------------------------------------------
    TSProblem(
        id="nextjs_api_route",
        category="nextjs",
        difficulty="easy",
        description="Basic Next.js API route handler (pages/api)",
        prompt=(
            "import type { NextApiRequest, NextApiResponse } from 'next';\n"
            "\n"
            "interface HelloResponse {\n"
            "  message: string;\n"
            "}\n"
            "\n"
            "/**\n"
            " * GET /api/hello — returns a greeting JSON response.\n"
            " */\n"
            "export default function handler(\n"
            "  req: NextApiRequest,\n"
            "  res: NextApiResponse<HelloResponse>,\n"
            "): void {"
        ),
        canonical_solution=(
            "export default function handler(\n"
            "  req: NextApiRequest,\n"
            "  res: NextApiResponse<HelloResponse>,\n"
            "): void {\n"
            "  if (req.method !== 'GET') {\n"
            "    res.status(405).json({ message: 'Method Not Allowed' });\n"
            "    return;\n"
            "  }\n"
            "  res.status(200).json({ message: 'Hello, World!' });\n"
            "}"
        ),
        test_code=(
            "test('returns 200', async () => {\n"
            "  const { req, res } = createMocks({ method: 'GET' });\n"
            "  handler(req, res);\n"
            "  expect(res._getStatusCode()).toBe(200);\n"
            "});\n"
        ),
        required_patterns=[r"NextApiRequest", r"NextApiResponse", r"res\.status\(200\)"],
    ),
    TSProblem(
        id="nextjs_get_server_side_props",
        category="nextjs",
        difficulty="medium",
        description="getServerSideProps with typed return",
        prompt=(
            "import type { GetServerSideProps, InferGetServerSidePropsType } from 'next';\n"
            "\n"
            "interface Post {\n"
            "  id: number;\n"
            "  title: string;\n"
            "}\n"
            "\n"
            "/**\n"
            " * Fetches the post with the given id from the server and passes it as props.\n"
            " * Uses the `id` query param from the URL.\n"
            " */\n"
            "export const getServerSideProps: GetServerSideProps<{ post: Post }> = async (context) => {"
        ),
        canonical_solution=(
            "export const getServerSideProps: GetServerSideProps<{ post: Post }> = async (context) => {\n"
            "  const { id } = context.params ?? {};\n"
            "  const res = await fetch(`https://jsonplaceholder.typicode.com/posts/${id}`);\n"
            "  const post: Post = await res.json();\n"
            "  return { props: { post } };\n"
            "};"
        ),
        test_code=(
            "// Integration test — skipped in unit mode\n"
        ),
        required_patterns=[r"GetServerSideProps", r"context\.params", r"return\s*\{?\s*props"],
    ),
    TSProblem(
        id="nextjs_middleware",
        category="nextjs",
        difficulty="medium",
        description="Next.js middleware for auth checking",
        prompt=(
            "import { NextResponse } from 'next/server';\n"
            "import type { NextRequest } from 'next/server';\n"
            "\n"
            "/**\n"
            " * Middleware that redirects unauthenticated users to /login\n"
            " * for any route under /dashboard.\n"
            " */\n"
            "export function middleware(request: NextRequest): NextResponse {"
        ),
        canonical_solution=(
            "export function middleware(request: NextRequest): NextResponse {\n"
            "  const token = request.cookies.get('session')?.value;\n"
            "  if (!token && request.nextUrl.pathname.startsWith('/dashboard')) {\n"
            "    return NextResponse.redirect(new URL('/login', request.url));\n"
            "  }\n"
            "  return NextResponse.next();\n"
            "}"
        ),
        test_code=(
            "// Middleware tests require Next.js test utilities\n"
        ),
        required_patterns=[r"NextRequest", r"NextResponse", r"NextResponse\.redirect|NextResponse.redirect"],
    ),
    TSProblem(
        id="nextjs_dynamic_route",
        category="nextjs",
        difficulty="medium",
        description="Page component with dynamic route params",
        prompt=(
            "import type { GetStaticPaths, GetStaticProps, InferGetStaticPropsType } from 'next';\n"
            "\n"
            "/**\n"
            " * Static paths for /posts/[id].\n"
            " * Generates paths for post IDs 1-5.\n"
            " */\n"
            "export const getStaticPaths: GetStaticPaths = async () => {"
        ),
        canonical_solution=(
            "export const getStaticPaths: GetStaticPaths = async () => {\n"
            "  const paths = [1, 2, 3, 4, 5].map(id => ({ params: { id: String(id) } }));\n"
            "  return { paths, fallback: false };\n"
            "};"
        ),
        test_code=(
            "// Static path tests require Next.js build utilities\n"
        ),
        required_patterns=[r"GetStaticPaths", r"paths", r"fallback"],
    ),
    TSProblem(
        id="nextjs_app_route_handler",
        category="nextjs",
        difficulty="medium",
        description="Next.js App Router route handler (app/api)",
        prompt=(
            "import { NextResponse } from 'next/server';\n"
            "import type { NextRequest } from 'next/server';\n"
            "\n"
            "/**\n"
            " * POST /api/users — creates a new user from the request body.\n"
            " * Returns 201 with the created user or 400 if body is invalid.\n"
            " */\n"
            "export async function POST(request: NextRequest): Promise<NextResponse> {"
        ),
        canonical_solution=(
            "export async function POST(request: NextRequest): Promise<NextResponse> {\n"
            "  try {\n"
            "    const body = await request.json();\n"
            "    if (!body.name || !body.email) {\n"
            "      return NextResponse.json({ error: 'Missing fields' }, { status: 400 });\n"
            "    }\n"
            "    return NextResponse.json({ id: crypto.randomUUID(), ...body }, { status: 201 });\n"
            "  } catch {\n"
            "    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });\n"
            "  }\n"
            "}"
        ),
        test_code=(
            "// App Router handler tests\n"
        ),
        required_patterns=[r"NextRequest", r"NextResponse", r"request\.json\(\)"],
    ),
    TSProblem(
        id="nextjs_layout",
        category="nextjs",
        difficulty="easy",
        description="Root layout component for Next.js App Router",
        prompt=(
            "import type { ReactNode } from 'react';\n"
            "\n"
            "export const metadata = {\n"
            "  title: 'My App',\n"
            "  description: 'A Next.js application',\n"
            "};\n"
            "\n"
            "/**\n"
            " * Root layout wraps all pages with HTML shell and body.\n"
            " */\n"
            "export default function RootLayout({ children }: { children: ReactNode }) {"
        ),
        canonical_solution=(
            "export default function RootLayout({ children }: { children: ReactNode }) {\n"
            "  return (\n"
            "    <html lang='en'>\n"
            "      <body>{children}</body>\n"
            "    </html>\n"
            "  );\n"
            "}"
        ),
        test_code=("// Layout rendering test\n"),
        required_patterns=[r"RootLayout", r"children", r"<html|<body"],
    ),
    # -----------------------------------------------------------------------
    # PRISMA (6 problems)
    # -----------------------------------------------------------------------
    TSProblem(
        id="prisma_find_user",
        category="prisma",
        difficulty="easy",
        description="Prisma findUnique query for a user by ID",
        prompt=(
            "import { PrismaClient } from '@prisma/client';\n"
            "\n"
            "const prisma = new PrismaClient();\n"
            "\n"
            "/**\n"
            " * Finds a user by their ID. Returns null if not found.\n"
            " */\n"
            "async function findUserById(id: string) {"
        ),
        canonical_solution=(
            "async function findUserById(id: string) {\n"
            "  return prisma.user.findUnique({\n"
            "    where: { id },\n"
            "  });\n"
            "}"
        ),
        test_code=(
            "// Requires a test database — integration test only\n"
        ),
        required_patterns=[r"findUnique", r"where.*id|where:\s*\{", r"prisma\.user"],
    ),
    TSProblem(
        id="prisma_create_with_relation",
        category="prisma",
        difficulty="medium",
        description="Prisma create with nested relation",
        prompt=(
            "import { PrismaClient } from '@prisma/client';\n"
            "\n"
            "const prisma = new PrismaClient();\n"
            "\n"
            "/**\n"
            " * Creates a new post authored by an existing user.\n"
            " * `authorId` must reference an existing User record.\n"
            " */\n"
            "async function createPost(title: string, content: string, authorId: string) {"
        ),
        canonical_solution=(
            "async function createPost(title: string, content: string, authorId: string) {\n"
            "  return prisma.post.create({\n"
            "    data: {\n"
            "      title,\n"
            "      content,\n"
            "      author: { connect: { id: authorId } },\n"
            "    },\n"
            "  });\n"
            "}"
        ),
        test_code=("// Integration test only\n"),
        required_patterns=[r"prisma\.post\.create", r"connect.*id|connect:\s*\{"],
    ),
    TSProblem(
        id="prisma_find_with_include",
        category="prisma",
        difficulty="medium",
        description="Prisma findMany with include for relations",
        prompt=(
            "import { PrismaClient } from '@prisma/client';\n"
            "\n"
            "const prisma = new PrismaClient();\n"
            "\n"
            "/**\n"
            " * Fetches all posts, including the author's name and email.\n"
            " */\n"
            "async function getPostsWithAuthors() {"
        ),
        canonical_solution=(
            "async function getPostsWithAuthors() {\n"
            "  return prisma.post.findMany({\n"
            "    include: {\n"
            "      author: {\n"
            "        select: { name: true, email: true },\n"
            "      },\n"
            "    },\n"
            "  });\n"
            "}"
        ),
        test_code=("// Integration test only\n"),
        required_patterns=[r"findMany", r"include.*author|include:\s*\{", r"select"],
    ),
    TSProblem(
        id="prisma_update",
        category="prisma",
        difficulty="easy",
        description="Prisma update record by ID",
        prompt=(
            "import { PrismaClient } from '@prisma/client';\n"
            "\n"
            "const prisma = new PrismaClient();\n"
            "\n"
            "/**\n"
            " * Updates the title of a post by ID.\n"
            " */\n"
            "async function updatePostTitle(id: string, newTitle: string) {"
        ),
        canonical_solution=(
            "async function updatePostTitle(id: string, newTitle: string) {\n"
            "  return prisma.post.update({\n"
            "    where: { id },\n"
            "    data: { title: newTitle },\n"
            "  });\n"
            "}"
        ),
        test_code=("// Integration test only\n"),
        required_patterns=[r"prisma\.post\.update", r"where.*id|where:\s*\{", r"data.*title|data:\s*\{"],
    ),
    TSProblem(
        id="prisma_transaction",
        category="prisma",
        difficulty="hard",
        description="Prisma interactive transaction",
        prompt=(
            "import { PrismaClient } from '@prisma/client';\n"
            "\n"
            "const prisma = new PrismaClient();\n"
            "\n"
            "/**\n"
            " * Transfers `amount` from one account to another in a single transaction.\n"
            " * Throws if either account does not have sufficient funds.\n"
            " */\n"
            "async function transfer(fromId: string, toId: string, amount: number) {"
        ),
        canonical_solution=(
            "async function transfer(fromId: string, toId: string, amount: number) {\n"
            "  return prisma.$transaction(async (tx) => {\n"
            "    const from = await tx.account.update({\n"
            "      where: { id: fromId },\n"
            "      data: { balance: { decrement: amount } },\n"
            "    });\n"
            "    if (from.balance < 0) throw new Error('Insufficient funds');\n"
            "    return tx.account.update({\n"
            "      where: { id: toId },\n"
            "      data: { balance: { increment: amount } },\n"
            "    });\n"
            "  });\n"
            "}"
        ),
        test_code=("// Integration test only\n"),
        required_patterns=[r"\$transaction", r"decrement|increment"],
    ),
    TSProblem(
        id="prisma_delete",
        category="prisma",
        difficulty="easy",
        description="Prisma deleteMany with where clause",
        prompt=(
            "import { PrismaClient } from '@prisma/client';\n"
            "\n"
            "const prisma = new PrismaClient();\n"
            "\n"
            "/**\n"
            " * Deletes all posts by a given author ID.\n"
            " */\n"
            "async function deletePostsByAuthor(authorId: string) {"
        ),
        canonical_solution=(
            "async function deletePostsByAuthor(authorId: string) {\n"
            "  return prisma.post.deleteMany({\n"
            "    where: { authorId },\n"
            "  });\n"
            "}"
        ),
        test_code=("// Integration test only\n"),
        required_patterns=[r"deleteMany", r"where.*authorId|where:\s*\{"],
    ),
    # -----------------------------------------------------------------------
    # ZOD (6 problems)
    # -----------------------------------------------------------------------
    TSProblem(
        id="zod_basic_schema",
        category="zod",
        difficulty="easy",
        description="Basic Zod schema for a user object",
        prompt=(
            "import { z } from 'zod';\n"
            "\n"
            "/**\n"
            " * Zod schema for a user with name, email, and optional age.\n"
            " */\n"
            "export const UserSchema = z.object({"
        ),
        canonical_solution=(
            "export const UserSchema = z.object({\n"
            "  name: z.string().min(1),\n"
            "  email: z.string().email(),\n"
            "  age: z.number().int().positive().optional(),\n"
            "});\n"
            "\n"
            "export type User = z.infer<typeof UserSchema>;"
        ),
        test_code=(
            "test('validates user', () => {\n"
            "  expect(() => UserSchema.parse({ name: 'Alice', email: 'a@b.com' })).not.toThrow();\n"
            "});\n"
        ),
        required_patterns=[r"z\.object", r"z\.string\(\)", r"z\.string\(\)\.email\(\)|email"],
    ),
    TSProblem(
        id="zod_validation_error",
        category="zod",
        difficulty="medium",
        description="Safe parse with error handling",
        prompt=(
            "import { z } from 'zod';\n"
            "\n"
            "const LoginSchema = z.object({\n"
            "  email: z.string().email(),\n"
            "  password: z.string().min(8),\n"
            "});\n"
            "\n"
            "type LoginInput = z.infer<typeof LoginSchema>;\n"
            "\n"
            "/**\n"
            " * Validates login input. Returns { success: true, data } or\n"
            " * { success: false, errors: string[] }.\n"
            " */\n"
            "function validateLogin(input: unknown): { success: true; data: LoginInput } | { success: false; errors: string[] } {"
        ),
        canonical_solution=(
            "function validateLogin(input: unknown): { success: true; data: LoginInput } | { success: false; errors: string[] } {\n"
            "  const result = LoginSchema.safeParse(input);\n"
            "  if (!result.success) {\n"
            "    return { success: false, errors: result.error.errors.map(e => e.message) };\n"
            "  }\n"
            "  return { success: true, data: result.data };\n"
            "}"
        ),
        test_code=(
            "test('returns errors for invalid input', () => {\n"
            "  const result = validateLogin({ email: 'bad', password: 'short' });\n"
            "  expect(result.success).toBe(false);\n"
            "});\n"
        ),
        required_patterns=[r"safeParse", r"result\.success", r"result\.error"],
    ),
    TSProblem(
        id="zod_schema_merge",
        category="zod",
        difficulty="medium",
        description="Merge two Zod schemas with .merge()",
        prompt=(
            "import { z } from 'zod';\n"
            "\n"
            "const BaseSchema = z.object({\n"
            "  id: z.string().uuid(),\n"
            "  createdAt: z.date(),\n"
            "});\n"
            "\n"
            "const PostSchema = z.object({\n"
            "  title: z.string().min(1).max(200),\n"
            "  body: z.string(),\n"
            "});\n"
            "\n"
            "/**\n"
            " * Schema for a post that has been persisted (includes base fields).\n"
            " * Merge BaseSchema and PostSchema.\n"
            " */\n"
            "export const PersistedPostSchema ="
        ),
        canonical_solution=(
            "export const PersistedPostSchema = BaseSchema.merge(PostSchema);\n"
            "export type PersistedPost = z.infer<typeof PersistedPostSchema>;"
        ),
        test_code=(
            "test('PersistedPostSchema has all fields', () => {\n"
            "  const shape = PersistedPostSchema.shape;\n"
            "  expect(shape).toHaveProperty('id');\n"
            "  expect(shape).toHaveProperty('title');\n"
            "});\n"
        ),
        required_patterns=[r"\.merge\(", r"PersistedPostSchema"],
    ),
    TSProblem(
        id="zod_schema_extend",
        category="zod",
        difficulty="medium",
        description="Extend a Zod schema with additional fields",
        prompt=(
            "import { z } from 'zod';\n"
            "\n"
            "const UserSchema = z.object({\n"
            "  id: z.string(),\n"
            "  name: z.string(),\n"
            "});\n"
            "\n"
            "/**\n"
            " * Admin user schema — extends UserSchema with a `role` field.\n"
            " */\n"
            "export const AdminSchema ="
        ),
        canonical_solution=(
            "export const AdminSchema = UserSchema.extend({\n"
            "  role: z.literal('admin'),\n"
            "  permissions: z.array(z.string()),\n"
            "});"
        ),
        test_code=(
            "test('AdminSchema requires role', () => {\n"
            "  expect(() => AdminSchema.parse({ id: '1', name: 'Alice', role: 'admin', permissions: [] })).not.toThrow();\n"
            "});\n"
        ),
        required_patterns=[r"\.extend\(", r"AdminSchema", r"z\.literal\('admin'\)|z.literal"],
    ),
    TSProblem(
        id="zod_enum",
        category="zod",
        difficulty="easy",
        description="Zod enum schema for status values",
        prompt=(
            "import { z } from 'zod';\n"
            "\n"
            "/**\n"
            " * Schema for order status — must be one of the allowed string values.\n"
            " */\n"
            "export const OrderStatusSchema = z.enum(["
        ),
        canonical_solution=(
            "export const OrderStatusSchema = z.enum(['pending', 'processing', 'shipped', 'delivered', 'cancelled']);\n"
            "export type OrderStatus = z.infer<typeof OrderStatusSchema>;"
        ),
        test_code=(
            "test('valid status', () => {\n"
            "  expect(() => OrderStatusSchema.parse('pending')).not.toThrow();\n"
            "  expect(() => OrderStatusSchema.parse('unknown')).toThrow();\n"
            "});\n"
        ),
        required_patterns=[r"z\.enum\(", r"OrderStatusSchema"],
    ),
    TSProblem(
        id="zod_transform",
        category="zod",
        difficulty="medium",
        description="Zod schema with transform to normalise input",
        prompt=(
            "import { z } from 'zod';\n"
            "\n"
            "/**\n"
            " * Schema for an email input that trims whitespace and lowercases the value.\n"
            " */\n"
            "export const EmailSchema ="
        ),
        canonical_solution=(
            "export const EmailSchema = z\n"
            "  .string()\n"
            "  .trim()\n"
            "  .toLowerCase()\n"
            "  .email();"
        ),
        test_code=(
            "test('normalises email', () => {\n"
            "  expect(EmailSchema.parse('  ALICE@EXAMPLE.COM  ')).toBe('alice@example.com');\n"
            "});\n"
        ),
        required_patterns=[r"EmailSchema", r"z\.string\(\)", r"\.email\(\)"],
    ),
    # -----------------------------------------------------------------------
    # TESTING (6 problems)
    # -----------------------------------------------------------------------
    TSProblem(
        id="testing_basic_jest",
        category="testing",
        difficulty="easy",
        description="Basic Jest test suite for a pure function",
        prompt=(
            "import { sum } from './sum';\n"
            "\n"
            "/**\n"
            " * Test suite for the `sum` function.\n"
            " */\n"
            "describe('sum', () => {"
        ),
        canonical_solution=(
            "describe('sum', () => {\n"
            "  it('adds two positive numbers', () => {\n"
            "    expect(sum(1, 2)).toBe(3);\n"
            "  });\n"
            "\n"
            "  it('handles zero', () => {\n"
            "    expect(sum(0, 0)).toBe(0);\n"
            "  });\n"
            "\n"
            "  it('handles negative numbers', () => {\n"
            "    expect(sum(-1, -2)).toBe(-3);\n"
            "  });\n"
            "});"
        ),
        test_code=("// Self-referential: this IS the test\n"),
        required_patterns=[r"describe\(", r"it\(|test\(", r"expect\(", r"\.toBe\("],
    ),
    TSProblem(
        id="testing_mock_function",
        category="testing",
        difficulty="medium",
        description="Create a Jest mock for a dependency",
        prompt=(
            "import { fetchUser } from './api';\n"
            "import { UserService } from './userService';\n"
            "\n"
            "jest.mock('./api');\n"
            "\n"
            "const mockFetchUser = fetchUser as jest.MockedFunction<typeof fetchUser>;\n"
            "\n"
            "/**\n"
            " * Tests for UserService.getUser — mocks the fetchUser API call.\n"
            " */\n"
            "describe('UserService.getUser', () => {"
        ),
        canonical_solution=(
            "describe('UserService.getUser', () => {\n"
            "  beforeEach(() => {\n"
            "    jest.clearAllMocks();\n"
            "  });\n"
            "\n"
            "  it('returns user from api', async () => {\n"
            "    mockFetchUser.mockResolvedValue({ id: '1', name: 'Alice' });\n"
            "    const service = new UserService();\n"
            "    const user = await service.getUser('1');\n"
            "    expect(user.name).toBe('Alice');\n"
            "    expect(mockFetchUser).toHaveBeenCalledWith('1');\n"
            "  });\n"
            "});"
        ),
        test_code=("// Self-referential\n"),
        required_patterns=[r"mockResolvedValue|mockReturnValue", r"toHaveBeenCalledWith", r"jest\.clearAllMocks\(\)|clearAllMocks"],
    ),
    TSProblem(
        id="testing_async_test",
        category="testing",
        difficulty="medium",
        description="Async test with async/await pattern",
        prompt=(
            "import { fetchPosts } from './api';\n"
            "\n"
            "/**\n"
            " * Tests for the fetchPosts function.\n"
            " * Uses async/await and mocks fetch.\n"
            " */\n"
            "describe('fetchPosts', () => {"
        ),
        canonical_solution=(
            "describe('fetchPosts', () => {\n"
            "  beforeEach(() => {\n"
            "    global.fetch = jest.fn();\n"
            "  });\n"
            "\n"
            "  it('returns parsed posts on success', async () => {\n"
            "    (global.fetch as jest.Mock).mockResolvedValue({\n"
            "      ok: true,\n"
            "      json: async () => [{ id: 1, title: 'Test' }],\n"
            "    });\n"
            "    const posts = await fetchPosts();\n"
            "    expect(posts).toHaveLength(1);\n"
            "    expect(posts[0].title).toBe('Test');\n"
            "  });\n"
            "\n"
            "  it('throws on fetch failure', async () => {\n"
            "    (global.fetch as jest.Mock).mockResolvedValue({ ok: false, status: 500 });\n"
            "    await expect(fetchPosts()).rejects.toThrow();\n"
            "  });\n"
            "});"
        ),
        test_code=("// Self-referential\n"),
        required_patterns=[r"async.*\(\)", r"await.*fetchPosts\(\)|await fetchPosts", r"rejects\.toThrow"],
    ),
    TSProblem(
        id="testing_spy",
        category="testing",
        difficulty="medium",
        description="Jest spy on a class method",
        prompt=(
            "import { Logger } from './logger';\n"
            "import { processData } from './processor';\n"
            "\n"
            "/**\n"
            " * Tests that processData calls logger.log with the correct arguments.\n"
            " */\n"
            "describe('processData', () => {"
        ),
        canonical_solution=(
            "describe('processData', () => {\n"
            "  it('logs processed data', () => {\n"
            "    const logger = new Logger();\n"
            "    const logSpy = jest.spyOn(logger, 'log');\n"
            "    processData([1, 2, 3], logger);\n"
            "    expect(logSpy).toHaveBeenCalledWith('Processed 3 items');\n"
            "  });\n"
            "});"
        ),
        test_code=("// Self-referential\n"),
        required_patterns=[r"jest\.spyOn", r"toHaveBeenCalledWith"],
    ),
    TSProblem(
        id="testing_before_after",
        category="testing",
        difficulty="easy",
        description="Test with beforeEach and afterEach lifecycle hooks",
        prompt=(
            "import { Database } from './db';\n"
            "\n"
            "/**\n"
            " * Tests for Database operations with proper setup and teardown.\n"
            " */\n"
            "describe('Database', () => {"
        ),
        canonical_solution=(
            "describe('Database', () => {\n"
            "  let db: Database;\n"
            "\n"
            "  beforeEach(() => {\n"
            "    db = new Database(':memory:');\n"
            "    db.connect();\n"
            "  });\n"
            "\n"
            "  afterEach(() => {\n"
            "    db.disconnect();\n"
            "  });\n"
            "\n"
            "  it('can insert a record', () => {\n"
            "    expect(db.insert({ id: 1, name: 'test' })).toBe(true);\n"
            "  });\n"
            "});"
        ),
        test_code=("// Self-referential\n"),
        required_patterns=[r"beforeEach", r"afterEach", r"describe\("],
    ),
    TSProblem(
        id="testing_snapshot",
        category="testing",
        difficulty="easy",
        description="React Testing Library snapshot test",
        prompt=(
            "import React from 'react';\n"
            "import { render } from '@testing-library/react';\n"
            "import { Button } from './Button';\n"
            "\n"
            "/**\n"
            " * Snapshot tests for the Button component.\n"
            " */\n"
            "describe('Button', () => {"
        ),
        canonical_solution=(
            "describe('Button', () => {\n"
            "  it('matches snapshot for primary variant', () => {\n"
            "    const { container } = render(\n"
            "      <Button label='Click me' onClick={jest.fn()} variant='primary' />\n"
            "    );\n"
            "    expect(container).toMatchSnapshot();\n"
            "  });\n"
            "\n"
            "  it('matches snapshot for secondary variant', () => {\n"
            "    const { container } = render(\n"
            "      <Button label='Cancel' onClick={jest.fn()} variant='secondary' />\n"
            "    );\n"
            "    expect(container).toMatchSnapshot();\n"
            "  });\n"
            "});"
        ),
        test_code=("// Self-referential\n"),
        required_patterns=[r"toMatchSnapshot\(\)", r"render\(", r"describe\("],
    ),
]

# Verify we have 50 problems
assert len(PROBLEMS) == 50, f"Expected 50 problems, got {len(PROBLEMS)}"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _structural_check(solution: str, problem: TSProblem) -> bool:
    """Check that the function/class name from the problem prompt appears in the solution."""
    # Strip JSDoc / block comments from prompt before extracting names so we
    # don't accidentally match words inside comment text (e.g. "type that …").
    prompt_no_comments = re.sub(r"/\*.*?\*/", "", problem.prompt, flags=re.DOTALL)
    prompt_no_comments = re.sub(r"//[^\n]*", "", prompt_no_comments)

    # Patterns that appear at statement level (start-of-line or after export)
    name_patterns = [
        r"(?:^|\n|;)\s*(?:export\s+(?:default\s+)?)?function\s+(\w+)",
        r"(?:^|\n|;)\s*(?:export\s+)?const\s+(\w+)",
        r"(?:^|\n|;)\s*(?:export\s+)?class\s+(\w+)",
        r"(?:^|\n|;)\s*(?:export\s+)?type\s+(\w+)",
        r"(?:^|\n|;)\s*(?:export\s+)?interface\s+(\w+)",
    ]
    for pat in name_patterns:
        m = re.search(pat, prompt_no_comments)
        if m:
            name = m.group(1)
            if name in solution:
                return True
    # Fall back: check at least one TypeScript annotation
    return bool(re.search(r":\s*(?:string|number|boolean|void|Promise|Record|Array)", solution))


def _type_annotation_check(solution: str) -> bool:
    """Check that the solution contains at least one TypeScript type annotation."""
    ts_patterns = [
        r":\s*string",
        r":\s*number",
        r":\s*boolean",
        r":\s*void",
        r":\s*Promise<",
        r":\s*\w+\[\]",
        r"<\w+>",
        r"interface\s+\w+",
        r"type\s+\w+\s*=",
    ]
    return any(re.search(p, solution) for p in ts_patterns)


def _pattern_check(solution: str, problem: TSProblem) -> bool:
    """Check that all required patterns appear and no forbidden patterns appear."""
    for pat in problem.required_patterns:
        if not re.search(pat, solution):
            return False
    for pat in problem.forbidden_patterns:
        if re.search(pat, solution):
            return False
    return True


def _tsc_check(solution: str, problem: TSProblem) -> bool | None:
    """Attempt tsc type-check if tsc is available.  Returns None if tsc not found."""
    if shutil.which("tsc") is None:
        return None

    # Write a minimal .ts file combining prompt + solution
    code = problem.prompt + "\n" + solution
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ts",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        tmp_path = Path(f.name)

    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--strict", "--target", "ESNext", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------


class TSBenchmark:
    """TypeScript-specific code generation benchmark.

    Evaluation is purely static — no TypeScript / Node.js runtime required.
    """

    PROBLEMS: list[TSProblem] = PROBLEMS

    def __init__(self, categories: list[str] | None = None) -> None:
        self._categories = categories

    def get_problems(self) -> list[TSProblem]:
        """Return problems, optionally filtered by category."""
        if not self._categories:
            return list(self.PROBLEMS)
        return [p for p in self.PROBLEMS if p.category in self._categories]

    def evaluate_solution(self, problem: TSProblem, solution: str) -> bool:
        """Evaluate a generated solution using a tiered static strategy.

        Returns True if the solution passes all tiers.
        """
        # Tier 1: structural presence of key identifiers
        if not _structural_check(solution, problem):
            return False

        # Tier 2: at least one TypeScript annotation
        if not _type_annotation_check(solution):
            return False

        # Tier 3: domain-specific pattern matching
        if not _pattern_check(solution, problem):
            return False

        # Tier 4: optional tsc check (bonus, doesn't fail if tsc is absent)
        tsc_result = _tsc_check(solution, problem)
        if tsc_result is False:
            return False

        return True

    def run(
        self,
        generator,  # cola_coder.inference.generator.CodeGenerator (or mock)
        tokenizer,  # unused — kept for API symmetry with HumanEval runner
        num_samples: int = 1,
        temperature: float = 0.2,
    ) -> "TSBenchmarkResult":
        """Run the full benchmark against a generator."""
        problems = self.get_problems()
        details: list[dict] = []
        solved = 0

        category_totals: dict[str, int] = {}
        category_solved: dict[str, int] = {}
        difficulty_totals: dict[str, int] = {}
        difficulty_solved: dict[str, int] = {}

        for problem in problems:
            category_totals[problem.category] = category_totals.get(problem.category, 0) + 1
            difficulty_totals[problem.difficulty] = difficulty_totals.get(problem.difficulty, 0) + 1

            num_correct = 0
            for _ in range(num_samples):
                try:
                    generated = generator.generate(
                        prompt=problem.prompt,
                        max_new_tokens=256,
                        temperature=temperature,
                        top_k=50,
                        top_p=0.9,
                    )
                    if self.evaluate_solution(problem, generated):
                        num_correct += 1
                except Exception:
                    pass

            passed = num_correct > 0
            if passed:
                solved += 1
                category_solved[problem.category] = category_solved.get(problem.category, 0) + 1
                difficulty_solved[problem.difficulty] = difficulty_solved.get(problem.difficulty, 0) + 1

            details.append(
                {
                    "id": problem.id,
                    "category": problem.category,
                    "difficulty": problem.difficulty,
                    "passed": passed,
                    "num_correct": num_correct,
                    "num_samples": num_samples,
                }
            )

        by_category = {
            cat: category_solved.get(cat, 0) / total
            for cat, total in category_totals.items()
        }
        by_difficulty = {
            diff: difficulty_solved.get(diff, 0) / total
            for diff, total in difficulty_totals.items()
        }

        return TSBenchmarkResult(
            total_problems=len(problems),
            solved=solved,
            pass_rate=solved / len(problems) if problems else 0.0,
            by_category=by_category,
            by_difficulty=by_difficulty,
            details=details,
        )
