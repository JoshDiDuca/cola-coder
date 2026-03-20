# Research: Dependency Graph-Aware Training

## Status: Novel — Academic concept, never implemented for code generation LLMs

## The Problem

Current code models see one file at a time. But real code is a WEB of dependencies:

```
app/page.tsx
  imports → components/Header.tsx
  imports → lib/api.ts
    imports → lib/types.ts
    imports → lib/auth.ts
      imports → lib/types.ts (shared dependency!)
```

When a developer writes `app/page.tsx`, they have all this context in their head.
The model sees NONE of it. It's like reading one chapter of a book with no idea
what happened in the other chapters.

## The Idea: Project-Level Training Examples

Instead of training on individual files, train on DEPENDENCY-ORDERED FILE GROUPS:

```
Training example = [types.ts] + [auth.ts] + [api.ts] + [page.tsx]
                    ← dependency order: leaves first, dependents last →
```

The model sees the type definitions BEFORE the code that uses them. It learns:
- How types flow through a codebase
- How abstractions are built on top of each other
- How to use APIs that were defined earlier in the context

### Concrete Example

Standard training (current):
```
<file>
import { User } from '../types';
import { fetchUser } from '../api';

export function UserProfile({ id }: { id: string }) {
  // Model has NO IDEA what User looks like or what fetchUser returns
  const user = await fetchUser(id);
  return <div>{user.name}</div>;
}
</file>
```

Dependency-aware training (proposed):
```
<file path="lib/types.ts">
export interface User {
  id: string;
  name: string;
  email: string;
  avatar: string;
}
</file>
<file path="lib/api.ts" depends_on="lib/types.ts">
import { User } from './types';

export async function fetchUser(id: string): Promise<User> {
  const res = await fetch(`/api/users/${id}`);
  return res.json();
}
</file>
<file path="components/UserProfile.tsx" depends_on="lib/types.ts,lib/api.ts">
import { User } from '../types';
import { fetchUser } from '../api';

export function UserProfile({ id }: { id: string }) {
  // NOW the model has seen User and fetchUser — it knows the types!
  const user = await fetchUser(id);
  return <div>{user.name}</div>;
}
</file>
```

## Implementation

### Step 1: Build Import Graph at Data Prep Time

```python
class ImportGraphBuilder:
    """Extract import/dependency graph from a repository.

    For TypeScript:
    - Parse import statements: import { X } from './module'
    - Resolve relative paths to actual files
    - Build directed graph: file → files it imports

    For Python:
    - Parse import statements: from module import X
    - Resolve to actual files (harder due to __init__.py, packages)
    """

    def build_graph(self, repo_path: Path, language: str) -> dict[str, list[str]]:
        """Returns adjacency list: {file: [dependencies]}"""
        graph = {}
        for file in self._find_code_files(repo_path, language):
            imports = self._extract_imports(file, language)
            resolved = [self._resolve_import(imp, file, repo_path) for imp in imports]
            graph[str(file)] = [r for r in resolved if r is not None]
        return graph

    def topological_sort(self, graph: dict) -> list[list[str]]:
        """Sort files in dependency order (leaves first).

        Returns groups: each group can be processed in parallel,
        but groups must be ordered. Like build system dependency resolution.
        """
```

### Step 2: Create Multi-File Training Examples

```python
class ProjectChunker:
    """Create training examples from dependency-ordered file groups.

    Strategy:
    1. Build import graph for the repo
    2. For each "entry point" file (component, API route, main):
       a. Walk its dependency tree (BFS/DFS)
       b. Topologically sort the dependencies
       c. Concatenate files in order: deps first, entry point last
       d. If total tokens > max_seq_len: truncate from the FRONT
          (keep the entry point, sacrifice deep dependencies)
       e. This becomes one training example

    Special tokens:
    <file path="..."> content </file>    — file boundaries
    <depends_on> path1, path2 </depends_on>  — dependency declaration
    """

    def chunk_repo(self, repo_path: Path, language: str) -> list[str]:
        graph = self.graph_builder.build_graph(repo_path, language)
        entry_points = self._find_entry_points(graph)

        chunks = []
        for entry in entry_points:
            deps = self._get_ordered_deps(graph, entry)
            chunk = self._format_multi_file(deps + [entry], repo_path)
            if len(self.tokenizer.encode(chunk)) <= self.max_seq_len:
                chunks.append(chunk)
            else:
                # Truncate from front, keep entry point
                chunks.append(self._truncate_chunk(chunk))
        return chunks
```

### Step 3: Mixed Training

Train on a MIX of single-file and multi-file examples:

```
70% single-file examples (standard, fast)
30% multi-file dependency-ordered examples (rich context)
```

The model learns both isolated code patterns AND cross-file relationships.

## Expected Benefits

1. **Better type inference**: Model has seen the type definitions in context
2. **Better API usage**: Model knows function signatures before using them
3. **Better imports**: Model learns which imports are needed for which APIs
4. **Project coherence**: Generated code is more likely to be consistent with
   the rest of the codebase (because it trained on coherent codebases)

## Challenges

1. **Sequence length**: Multi-file examples are longer. Need 4K-8K context for
   meaningful dependency chains. Current tiny model has 1024 → need larger model.
2. **Repo cloning**: Need to clone repos to build import graphs. Can't do this
   from StarCoderData alone (it's individual files, no repo structure).
3. **Import resolution**: TypeScript's path resolution (tsconfig paths, node_modules,
   barrel exports) is complex. Need a simplified resolver.
4. **Training data size**: Each multi-file example uses more tokens per "concept"
   than a single-file example. Need more total tokens.

## Feasibility

| Aspect | Assessment |
|--------|-----------|
| Novelty | Very high — true project-level training is unexplored |
| Difficulty | High (import resolution, graph building, chunking) |
| Expected impact | Very high for multi-file generation tasks |
| Prerequisites | GitHub scraper (plan 03), larger context window |
| Risk | Medium — import resolution bugs could create broken examples |

## Prior Art

- RepoCoder (2023): Retrieves relevant files at inference time, not training time
- RepoFusion (2023): Concatenates related files, but no dependency ordering
- SWE-bench (2023): Evaluates on repo-level tasks, but models are standard file-level
- **Nobody has used topologically-sorted dependency graphs as training sequences**

## Connection to Multi-Agent Vision

This directly supports the multi-agent architecture:
- Router model: sees the full import graph, decides which specialist to invoke
- Specialist model: sees dependency-ordered context for its domain
- Example: React specialist sees `types.ts → hooks.ts → Component.tsx` in order
