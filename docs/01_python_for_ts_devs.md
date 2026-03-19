# Python for TypeScript Developers

A practical translation guide for experienced TS developers moving to Python for ML/AI work.
No hand-holding on programming basics -- just the mapping from what you know to what you need.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Basic Syntax](#2-basic-syntax)
3. [Types](#3-types)
4. [Variables & Data Structures](#4-variables--data-structures)
5. [Functions](#5-functions)
6. [Classes](#6-classes)
7. [Modules & Imports](#7-modules--imports)
8. [Iteration](#8-iteration)
9. [Error Handling](#9-error-handling)
10. [String Formatting](#10-string-formatting)
11. [File I/O](#11-file-io)
12. [Key Python-Specific Patterns in ML Code](#12-key-python-specific-patterns-in-ml-code)
13. [Package Management](#13-package-management)
14. [Testing](#14-testing)
15. [Common Gotchas](#15-common-gotchas)

---

## 1. Getting Started

### Virtual Environments (venv) -- like node_modules, but different

In Node, `npm install` dumps packages into `node_modules/` in your project. Python has
a similar concept called a **virtual environment** (venv), but it works differently: it
creates an isolated Python installation rather than a project-local folder of packages.

```bash
# Create a virtual environment (do this once per project)
python -m venv .venv

# Activate it (do this every time you open a new terminal)
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Now `python` and `pip` point to the venv's copies
# Deactivate when you're done:
deactivate
```

Key differences from Node:
- You have to **activate** the venv in each terminal session. There is no automatic detection.
- Packages install into the venv globally (no nested `node_modules` trees).
- The `.venv/` directory is always gitignored. It is not committed.

### pip -- like npm

```bash
# npm install torch        →  pip install torch
# npm install              →  pip install -r requirements.txt
# npm install --save torch →  pip install torch && pip freeze > requirements.txt
# npx                      →  python -m <module>
```

### Running Scripts

```bash
# node script.js           →  python script.py
# node -e "console.log(1)" →  python -c "print(1)"
# ts-node script.ts        →  python script.py  (no compile step needed)
```

Python files are just `.py` -- no build step, no transpilation, no `tsconfig.json`.

---

## 2. Basic Syntax

### Indentation Instead of Braces

This is the big one. Python uses indentation (4 spaces by convention) to define blocks.
No braces, no semicolons.

**TypeScript:**
```typescript
function greet(name: string): string {
    if (name === "world") {
        return `Hello, ${name}!`;
    } else {
        return `Hi, ${name}.`;
    }
}
```

**Python:**
```python
def greet(name: str) -> str:
    if name == "world":
        return f"Hello, {name}!"
    else:
        return f"Hi, {name}."
```

Things to notice:
- Colons (`:`) end statements that start a block (`if`, `def`, `for`, `class`, `with`, etc.).
- No parentheses around `if` conditions (you *can* use them, but it is not idiomatic).
- `===` does not exist. Python uses `==` for equality (it is type-safe enough in practice).
- No semicolons. Ever. If you add them, Python won't complain, but people will look at you funny.

### Comments

```typescript
// single line comment
/* multi-line
   comment */
```

```python
# single line comment

# Python has no multi-line comment syntax.
# You just use multiple # lines.

"""
Docstrings (triple-quoted strings) are used for documentation,
not general comments. They go at the top of modules, classes, and functions.
"""
```

### Line Continuation

```python
# Long lines can be broken with a backslash:
total = first_value + \
        second_value

# But anything inside brackets auto-continues:
result = (
    first_value
    + second_value
    + third_value
)

my_list = [
    1, 2, 3,
    4, 5, 6,
]
```

---

## 3. Types

Python has type hints that look a lot like TypeScript types. The critical difference:
**they are not enforced at runtime.** They are purely for tooling (mypy, pyright, your IDE).
You can lie in your type hints and Python will happily run the code.

### Basic Type Annotations

**TypeScript:**
```typescript
let name: string = "Alice";
let age: number = 30;
let active: boolean = true;
let scores: number[] = [1, 2, 3];
let data: Record<string, number> = { a: 1, b: 2 };
```

**Python:**
```python
name: str = "Alice"
age: int = 30
active: bool = True
scores: list[int] = [1, 2, 3]
data: dict[str, int] = {"a": 1, "b": 2}
```

Note: `list[int]` and `dict[str, int]` syntax requires Python 3.9+. In older code you'll
see `List[int]` and `Dict[str, int]` imported from `typing`. The lowercase versions are
preferred now.

### Common Type Mappings

| TypeScript | Python |
|---|---|
| `string` | `str` |
| `number` | `int` or `float` |
| `boolean` | `bool` |
| `null` / `undefined` | `None` |
| `any` | `Any` (from `typing`) |
| `unknown` | no direct equivalent |
| `void` | `-> None` |
| `never` | `NoReturn` (from `typing`) |
| `string \| number` | `str \| int` (3.10+) or `Union[str, int]` |
| `string \| null` | `str \| None` (3.10+) or `Optional[str]` |
| `Array<T>` | `list[T]` |
| `Record<K, V>` | `dict[K, V]` |
| `[string, number]` | `tuple[str, int]` |
| `Set<T>` | `set[T]` |

### Optional and Union

**TypeScript:**
```typescript
function find(id: number): string | null {
    // ...
}

function process(value: string | number): void {
    // ...
}
```

**Python:**
```python
from typing import Optional, Union  # only needed for older syntax

# Modern (3.10+):
def find(id: int) -> str | None:
    ...

def process(value: str | int) -> None:
    ...

# Older style (still common in codebases):
def find(id: int) -> Optional[str]:
    ...

def process(value: Union[str, int]) -> None:
    ...
```

### Generics

**TypeScript:**
```typescript
function first<T>(items: T[]): T | undefined {
    return items[0];
}
```

**Python:**
```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T | None:
    return items[0] if items else None
```

Python 3.12 introduced a cleaner syntax:

```python
# Python 3.12+
def first[T](items: list[T]) -> T | None:
    return items[0] if items else None
```

### Type Aliases

**TypeScript:**
```typescript
type Tensor = number[][];
type Config = { lr: number; epochs: number };
```

**Python:**
```python
# Simple alias
Tensor = list[list[float]]

# With TypeAlias for clarity (3.10+):
from typing import TypeAlias
Tensor: TypeAlias = list[list[float]]

# For the config, use TypedDict:
from typing import TypedDict

class Config(TypedDict):
    lr: float
    epochs: int
```

---

## 4. Variables & Data Structures

### Variables

**TypeScript:**
```typescript
let x = 10;       // mutable
const y = 20;     // immutable binding
var z = 30;       // don't use this
```

**Python:**
```python
x = 10             # all assignments are mutable
y = 20             # there is no const keyword
UPPER_CASE = 30    # convention for "constants" (not enforced)
```

Python has no `const`. If you want something to be treated as a constant, name it in
`UPPER_CASE`. The `Final` type hint exists but is not enforced at runtime:

```python
from typing import Final
MAX_EPOCHS: Final = 100  # type checkers will warn if you reassign this
```

### Lists (like Arrays)

**TypeScript:**
```typescript
const arr = [1, 2, 3];
arr.push(4);
arr.length;
arr.includes(2);
arr.slice(1, 3);
arr.indexOf(2);
const [first, ...rest] = arr;
```

**Python:**
```python
arr = [1, 2, 3]
arr.append(4)
len(arr)
2 in arr
arr[1:3]
arr.index(2)
first, *rest = arr
```

Common list operations you'll use in ML code:

```python
# Create a list of zeros
zeros = [0] * 10                    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# List comprehension (covered more in section 8)
squares = [x ** 2 for x in range(10)]

# Concatenation
combined = [1, 2] + [3, 4]          # [1, 2, 3, 4]

# Nested lists (poor man's matrix)
matrix = [[0] * 3 for _ in range(3)]
```

### Dicts (like Objects / Records)

**TypeScript:**
```typescript
const config = { lr: 0.001, epochs: 10, batchSize: 32 };
config.lr;
config["lr"];
Object.keys(config);
Object.values(config);
Object.entries(config);
const { lr, ...rest } = config;
"lr" in config;
```

**Python:**
```python
config = {"lr": 0.001, "epochs": 10, "batch_size": 32}
config["lr"]                      # KeyError if missing
config.get("lr")                  # None if missing
config.get("lr", 0.01)           # default if missing
config.keys()
config.values()
config.items()                    # like Object.entries()
lr = config.pop("lr")            # remove and return
"lr" in config

# No spread operator, but you can merge:
merged = {**config, "lr": 0.01}  # like { ...config, lr: 0.01 }
```

Important: Python dict keys are usually strings, but they can be any hashable type
(ints, tuples, etc.). You access them with brackets `config["lr"]`, not dots `config.lr`.
Dot access does not work on plain dicts.

### Tuples (no TS equivalent, closest is `as const` arrays)

Tuples are immutable sequences. You'll see them everywhere in ML code for shapes.

```python
shape = (3, 224, 224)          # a tuple
batch_shape = (32, *shape)     # unpacking into a new tuple: (32, 3, 224, 224)

# Single-element tuple needs a trailing comma:
single = (42,)                 # tuple
not_a_tuple = (42)             # just the int 42

# Tuples are immutable:
shape[0] = 5                   # TypeError!

# You'll see tuples as return values constantly:
def get_dimensions() -> tuple[int, int]:
    return 1920, 1080          # parentheses are optional
```

### Sets

**TypeScript:**
```typescript
const s = new Set([1, 2, 3]);
s.add(4);
s.has(2);
s.delete(2);
```

**Python:**
```python
s = {1, 2, 3}                 # literal syntax (not a dict -- no colons)
s.add(4)
2 in s
s.remove(2)                   # KeyError if missing
s.discard(2)                  # no error if missing

# Set operations:
a = {1, 2, 3}
b = {2, 3, 4}
a | b                         # union: {1, 2, 3, 4}
a & b                         # intersection: {2, 3}
a - b                         # difference: {1}
```

### None (like null)

Python has `None` instead of `null`/`undefined`. There is only one "nothing" value.

```python
x = None

# Check for None with `is`, not `==`:
if x is None:
    print("nothing here")

if x is not None:
    print("got something")
```

### Truthiness

Python's truthy/falsy rules are close to JS, but not identical:

```python
# Falsy values:
False, None, 0, 0.0, "", [], {}, set()

# Everything else is truthy.
# Unlike JS: there is no distinction between null and undefined.
# Unlike JS: empty collections are falsy (empty array [] is falsy in Python, truthy in JS).
```

---

## 5. Functions

### Basic Functions

**TypeScript:**
```typescript
function add(a: number, b: number): number {
    return a + b;
}

const add = (a: number, b: number): number => a + b;
```

**Python:**
```python
def add(a: int, b: int) -> int:
    return a + b

# Lambda (like arrow functions, but limited to a single expression):
add = lambda a, b: a + b
```

Lambdas in Python are intentionally limited. If you need multiple lines, use `def`.
You'll see lambdas mostly in `sort()` keys and small callbacks.

### Default Parameters

**TypeScript:**
```typescript
function train(epochs: number = 10, lr: number = 0.001): void {
    // ...
}
train(5);
train(5, 0.01);
```

**Python:**
```python
def train(epochs: int = 10, lr: float = 0.001) -> None:
    ...

train(5)
train(5, 0.01)
train(epochs=5, lr=0.01)       # keyword arguments -- very common in Python
train(lr=0.01, epochs=5)       # order doesn't matter with keyword args
```

Keyword arguments are huge in Python. Most ML APIs rely on them heavily:

```python
model = Transformer(
    d_model=512,
    n_heads=8,
    n_layers=6,
    dropout=0.1,
)
```

### *args and **kwargs (like rest params and spread)

**TypeScript:**
```typescript
function log(...args: any[]): void {
    console.log(...args);
}

function create(opts: { name: string; age?: number }): void {
    // ...
}
```

**Python:**
```python
def log(*args):
    """*args captures positional arguments as a tuple."""
    print(*args)

def create(**kwargs):
    """**kwargs captures keyword arguments as a dict."""
    name = kwargs["name"]
    age = kwargs.get("age")

# In practice, you'll see both together:
def flexible(*args, **kwargs):
    print(args)    # tuple of positional args
    print(kwargs)  # dict of keyword args

# Spreading into a function call:
params = {"d_model": 512, "n_heads": 8}
model = Transformer(**params)    # like Transformer({...params}) in spirit
```

### Keyword-Only Arguments

Python can force callers to use keyword syntax. You'll see this in ML APIs:

```python
# Everything after * must be passed as a keyword argument:
def train(dataset, *, epochs=10, lr=0.001):
    ...

train(my_data, epochs=5)       # ok
train(my_data, 5)              # TypeError!
```

---

## 6. Classes

### Basic Class

**TypeScript:**
```typescript
class Model {
    private name: string;
    public layers: number;

    constructor(name: string, layers: number = 6) {
        this.name = name;
        this.layers = layers;
    }

    forward(x: number[]): number[] {
        return x;
    }

    toString(): string {
        return `Model(${this.name})`;
    }
}
```

**Python:**
```python
class Model:
    def __init__(self, name: str, layers: int = 6):
        self._name = name          # _ prefix = "private by convention" (not enforced)
        self.layers = layers       # public

    def forward(self, x: list[float]) -> list[float]:
        return x

    def __repr__(self) -> str:    # like toString()
        return f"Model({self._name})"
```

Key differences:
- `self` is explicit. Every instance method takes `self` as its first parameter. It is
  like `this`, but you have to write it out.
- `__init__` is the constructor. The double underscores are called "dunders."
- No access modifiers (`public`, `private`, `protected`). Use `_single_underscore` for
  "private by convention" and `__double_underscore` for name-mangled (rarely used).
- No `new` keyword. You just call the class: `m = Model("gpt", 12)`.

### Inheritance

**TypeScript:**
```typescript
class TransformerModel extends Model {
    constructor(name: string, layers: number, heads: number) {
        super(name, layers);
        this.heads = heads;
    }
}
```

**Python:**
```python
class TransformerModel(Model):
    def __init__(self, name: str, layers: int, heads: int):
        super().__init__(name, layers)
        self.heads = heads
```

In PyTorch, you'll inherit from `nn.Module` constantly:

```python
import torch.nn as nn

class MyTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)
```

### @property (like getters/setters)

**TypeScript:**
```typescript
class Config {
    private _lr: number = 0.001;
    get lr(): number { return this._lr; }
    set lr(value: number) { this._lr = Math.max(0, value); }
}
```

**Python:**
```python
class Config:
    def __init__(self):
        self._lr = 0.001

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float):
        self._lr = max(0, value)

config = Config()
config.lr          # calls the getter
config.lr = 0.01   # calls the setter
```

### @staticmethod and @classmethod

**TypeScript:**
```typescript
class Model {
    static fromPretrained(path: string): Model {
        // ...
    }
    static defaultConfig(): Config {
        // ...
    }
}
```

**Python:**
```python
class Model:
    @classmethod
    def from_pretrained(cls, path: str) -> "Model":
        """cls is the class itself (like Model). Works with inheritance."""
        instance = cls()
        # load weights...
        return instance

    @staticmethod
    def default_config() -> dict:
        """No cls or self. Just a function namespaced to the class."""
        return {"lr": 0.001, "epochs": 10}
```

`@classmethod` receives the class as the first argument (`cls`). Use it for alternative
constructors (factory methods). `@staticmethod` receives nothing -- it is just a regular
function scoped to the class.

### Dataclasses (like interfaces with defaults)

If you miss TypeScript interfaces with default values, `dataclass` is your friend:

**TypeScript:**
```typescript
interface TrainConfig {
    lr: number;
    epochs: number;
    batchSize: number;
    dropout?: number;
}

const config: TrainConfig = {
    lr: 0.001,
    epochs: 10,
    batchSize: 32,
};
```

**Python:**
```python
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    lr: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    dropout: float = 0.1

    # For mutable defaults, use field():
    layer_sizes: list[int] = field(default_factory=lambda: [512, 256])

config = TrainConfig()                          # all defaults
config = TrainConfig(lr=0.01, epochs=20)        # override some
config.lr                                       # attribute access (dot notation!)
```

Dataclasses auto-generate `__init__`, `__repr__`, and `__eq__` for you. You'll see them
used heavily for configs in ML projects.

```python
# Frozen (immutable) dataclass:
@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
```

---

## 7. Modules & Imports

### Import Syntax

**TypeScript:**
```typescript
import torch from "torch";
import { Linear, Module } from "torch.nn";
import * as F from "torch.nn.functional";
import { readFile } from "fs/promises";
```

**Python:**
```python
import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
from pathlib import Path
```

Key differences:
- No `export` keyword. Everything at the top level of a `.py` file is importable by default.
- To make something "private" to a module, prefix it with `_` (convention).
- `import X` gives you the module. `from X import Y` gives you a specific name from it.
- `import X as Y` aliases the whole module. Very common: `import numpy as np`.

### Common ML Import Conventions

You'll see these in virtually every ML codebase:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
```

### `__init__.py` (like index.ts)

In TypeScript, `index.ts` re-exports from a directory. Python uses `__init__.py`:

```
# TypeScript                    # Python
src/                            src/
  models/                         models/
    index.ts                        __init__.py
    transformer.ts                  transformer.py
    attention.ts                    attention.py
```

```python
# models/__init__.py
from .transformer import Transformer
from .attention import MultiHeadAttention

# Now users can do:
from models import Transformer
```

Without `__init__.py`, a directory is not a package (in older Python). In modern Python
(3.3+), implicit namespace packages exist, but you should still use `__init__.py` for
clarity.

### Relative Imports

```python
# Inside models/transformer.py:
from .attention import MultiHeadAttention      # same directory
from ..utils import tokenize                   # parent directory
from . import config                           # import sibling module
```

Relative imports only work inside packages (directories with `__init__.py`).

---

## 8. Iteration

### For Loops

**TypeScript:**
```typescript
const items = [1, 2, 3];

// for...of
for (const item of items) {
    console.log(item);
}

// with index
items.forEach((item, i) => {
    console.log(i, item);
});

// traditional
for (let i = 0; i < 10; i++) {
    console.log(i);
}
```

**Python:**
```python
items = [1, 2, 3]

# direct iteration
for item in items:
    print(item)

# with index (like forEach with index)
for i, item in enumerate(items):
    print(i, item)

# range (like traditional for loop)
for i in range(10):            # 0 to 9
    print(i)

for i in range(2, 10):        # 2 to 9
for i in range(0, 10, 2):     # 0, 2, 4, 6, 8
```

### List Comprehensions (like .map() and .filter())

This is one of Python's best features. You'll use it constantly.

**TypeScript:**
```typescript
const squares = [1, 2, 3, 4, 5].map(x => x ** 2);
const evens = [1, 2, 3, 4, 5].filter(x => x % 2 === 0);
const evenSquares = [1, 2, 3, 4, 5]
    .filter(x => x % 2 === 0)
    .map(x => x ** 2);
```

**Python:**
```python
squares = [x ** 2 for x in [1, 2, 3, 4, 5]]
evens = [x for x in [1, 2, 3, 4, 5] if x % 2 == 0]
even_squares = [x ** 2 for x in [1, 2, 3, 4, 5] if x % 2 == 0]
```

The general pattern is: `[expression for item in iterable if condition]`

```python
# Dict comprehension:
word_to_id = {word: i for i, word in enumerate(vocab)}

# Set comprehension:
unique_lengths = {len(word) for word in words}

# Nested comprehension:
flat = [x for row in matrix for x in row]   # like matrix.flat() in JS
```

### zip (iterating multiple sequences together)

No direct TS equivalent. Like a multi-array `forEach`:

```python
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]

for name, score in zip(names, scores):
    print(f"{name}: {score}")

# zip stops at the shortest sequence. Use zip_longest for padding:
from itertools import zip_longest
```

### Useful Iteration Tools

```python
# any / all (like .some() / .every()):
any(x > 0 for x in values)    # True if any value is positive
all(x > 0 for x in values)    # True if all values are positive

# sum:
total = sum(x ** 2 for x in values)

# min / max with key:
longest = max(words, key=len)

# sorted:
sorted_words = sorted(words, key=len, reverse=True)

# reversed:
for item in reversed(items):
    print(item)
```

### While Loops

Same concept, different syntax:

```python
while loss > threshold:
    loss = train_step()
```

Python has no `do...while`. If you need it, use `while True` with a `break`:

```python
while True:
    result = process()
    if result.converged:
        break
```

---

## 9. Error Handling

### try/except (like try/catch)

**TypeScript:**
```typescript
try {
    const data = JSON.parse(text);
} catch (error) {
    if (error instanceof SyntaxError) {
        console.error("Bad JSON");
    } else {
        throw error;
    }
} finally {
    cleanup();
}
```

**Python:**
```python
import json

try:
    data = json.loads(text)
except json.JSONDecodeError:
    print("Bad JSON")
except Exception as e:
    raise                          # re-raise the current exception
finally:
    cleanup()
```

### Raising Exceptions (like throw)

**TypeScript:**
```typescript
throw new Error("something went wrong");
throw new TypeError("expected string");
```

**Python:**
```python
raise ValueError("something went wrong")
raise TypeError("expected string")

# Common exception types:
# ValueError    - wrong value (like "expected positive number")
# TypeError     - wrong type
# KeyError      - missing dict key
# IndexError    - list index out of range
# FileNotFoundError
# NotImplementedError  - for abstract methods
# RuntimeError  - general runtime error (common in PyTorch)
```

### Custom Exceptions

```python
class ModelNotFoundError(Exception):
    pass

class TrainingError(Exception):
    def __init__(self, epoch: int, message: str):
        self.epoch = epoch
        super().__init__(f"Epoch {epoch}: {message}")

raise TrainingError(5, "loss diverged")
```

### else Clause on try

Python has a unique `else` on try blocks -- it runs only if no exception was raised:

```python
try:
    result = compute()
except ComputeError:
    handle_error()
else:
    # only runs if no exception
    save(result)
finally:
    cleanup()
```

---

## 10. String Formatting

### f-strings (like template literals)

**TypeScript:**
```typescript
const name = "Alice";
const msg = `Hello, ${name}! Score: ${score.toFixed(2)}`;
console.log(`Epoch ${epoch}/${total}`);
```

**Python:**
```python
name = "Alice"
msg = f"Hello, {name}! Score: {score:.2f}"
print(f"Epoch {epoch}/{total}")
```

f-strings support format specifiers after a colon:

```python
f"{value:.4f}"       # 4 decimal places: "3.1416"
f"{value:>10}"       # right-align in 10 chars: "     hello"
f"{value:,}"         # thousands separator: "1,000,000"
f"{value:.2%}"       # percentage: "85.50%"
f"{value:#x}"        # hex: "0xff"

# Expressions inside f-strings:
f"{len(data)} items"
f"{'even' if x % 2 == 0 else 'odd'}"

# Self-documenting (Python 3.8+, great for debugging):
f"{variable=}"       # prints "variable=42"
f"{len(data)=}"      # prints "len(data)=100"
```

### Multiline Strings

**TypeScript:**
```typescript
const query = `
  SELECT *
  FROM users
  WHERE active = true
`;
```

**Python:**
```python
query = """
  SELECT *
  FROM users
  WHERE active = true
"""

# Or with f-string:
prompt = f"""
You are a helpful assistant.
The user's name is {name}.
"""
```

Triple quotes (`"""` or `'''`) are multiline strings. They preserve whitespace and
newlines exactly as written. Use `textwrap.dedent()` if you want to strip leading
indentation:

```python
from textwrap import dedent

query = dedent("""
    SELECT *
    FROM users
    WHERE active = true
""").strip()
```

---

## 11. File I/O

### The `with` Statement (Context Managers)

**TypeScript:**
```typescript
import { readFileSync, writeFileSync } from "fs";

const content = readFileSync("data.txt", "utf-8");
writeFileSync("output.txt", result);

// Or async:
const handle = await fs.open("data.txt");
try {
    const content = await handle.readFile("utf-8");
} finally {
    await handle.close();
}
```

**Python:**
```python
# Reading
with open("data.txt", "r") as f:
    content = f.read()

# Writing
with open("output.txt", "w") as f:
    f.write(result)

# The with statement automatically closes the file when the block exits,
# even if an exception occurs. It is like a built-in try/finally.

# Reading lines:
with open("data.txt") as f:
    lines = f.readlines()          # list of strings
    # or iterate line by line (memory-efficient):
    for line in f:
        process(line.strip())
```

### pathlib.Path (like Node's path module, but better)

**TypeScript:**
```typescript
import path from "path";
import { existsSync, mkdirSync } from "fs";

const dataDir = path.join(__dirname, "data");
if (!existsSync(dataDir)) {
    mkdirSync(dataDir, { recursive: true });
}
const files = readdirSync(dataDir).filter(f => f.endsWith(".json"));
```

**Python:**
```python
from pathlib import Path

data_dir = Path(__file__).parent / "data"     # / operator joins paths!
data_dir.mkdir(parents=True, exist_ok=True)
files = list(data_dir.glob("*.json"))

# Common Path operations:
p = Path("models/checkpoint.pt")
p.name              # "checkpoint.pt"
p.stem              # "checkpoint"
p.suffix             # ".pt"
p.parent             # Path("models")
p.exists()
p.is_file()
p.is_dir()
p.read_text()        # reads entire file as string
p.write_text(data)   # writes string to file
p.read_bytes()       # reads as bytes
p.resolve()          # absolute path

# Iterate over directory:
for f in Path("data").iterdir():
    print(f.name)
```

`pathlib.Path` is universally preferred in modern Python. You will see `os.path.join()`
in older code, but use `Path` for new code.

### JSON

**TypeScript:**
```typescript
const data = JSON.parse(text);
const text = JSON.stringify(data, null, 2);
```

**Python:**
```python
import json

data = json.loads(text)                          # parse string
text = json.dumps(data, indent=2)                # serialize to string

# File I/O:
with open("config.json") as f:
    config = json.load(f)                        # parse from file

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)               # write to file
```

---

## 12. Key Python-Specific Patterns in ML Code

### Decorators (@something)

Decorators wrap functions or classes. You know them from TS (experimental), but in Python
they are everywhere and stable.

**TypeScript (experimental):**
```typescript
function log(target: any, key: string, descriptor: PropertyDescriptor) {
    // ...
}

class Model {
    @log
    train() { /* ... */ }
}
```

**Python:**
```python
# A decorator is just a function that takes a function and returns a function.
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@timer
def train(epochs: int):
    ...

# @timer is syntax sugar for: train = timer(train)
```

Decorators you'll see constantly in ML code:

```python
@torch.no_grad()          # disable gradient computation (for inference)
def evaluate(model, data):
    ...

@torch.inference_mode()   # stricter version of no_grad
def predict(model, x):
    ...

@staticmethod
@classmethod
@property
@dataclass
@functools.lru_cache()    # memoization
@abstractmethod           # like abstract in TS
```

### Dunder Methods (\_\_magic\_\_)

Dunder ("double underscore") methods let you define how objects behave with built-in
operations. Think of them as implementing interfaces, but implicitly.

**TypeScript equivalent thinking:**
```typescript
class Vector {
    toString(): string { return `Vector(${this.x}, ${this.y})`; }
    valueOf(): number { return this.magnitude; }
    [Symbol.iterator]() { /* ... */ }
}
```

**Python:**
```python
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:           # like toString(), for debugging
        return f"Vector({self.x}, {self.y})"

    def __str__(self) -> str:            # human-readable string
        return f"({self.x}, {self.y})"

    def __len__(self) -> int:            # len(vec)
        return 2

    def __getitem__(self, i):            # vec[0], vec[1]
        return (self.x, self.y)[i]

    def __add__(self, other):            # vec1 + vec2
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other) -> bool:     # vec1 == vec2
        return self.x == other.x and self.y == other.y

    def __iter__(self):                  # for val in vec / unpacking
        yield self.x
        yield self.y

    def __call__(self, scale: float):    # vec(2.0) -- call it like a function
        return Vector(self.x * scale, self.y * scale)
```

The `__call__` dunder is important for PyTorch -- it is why you can call a model like a
function: `output = model(input)`. Under the hood, `nn.Module.__call__` invokes your
`forward()` method.

Common dunders you'll encounter:

| Dunder | Triggered by | TS-ish equivalent |
|---|---|---|
| `__init__` | `MyClass()` | `constructor` |
| `__repr__` | `repr(obj)`, debugger | `toString()` |
| `__str__` | `str(obj)`, `print(obj)` | `toString()` |
| `__len__` | `len(obj)` | `.length` |
| `__getitem__` | `obj[key]` | `[]` operator |
| `__setitem__` | `obj[key] = val` | `[]` assignment |
| `__contains__` | `x in obj` | `.includes()` |
| `__iter__` | `for x in obj` | `[Symbol.iterator]` |
| `__call__` | `obj()` | no equivalent |
| `__enter__`/`__exit__` | `with obj:` | no equivalent |
| `__add__` | `a + b` | operator overloading |

### Slicing

Python's slice syntax is more powerful than JS's `.slice()`:

```python
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

lst[2:5]          # [2, 3, 4]       -- start:stop (stop is exclusive)
lst[:3]           # [0, 1, 2]       -- first 3
lst[-3:]          # [7, 8, 9]       -- last 3
lst[::2]          # [0, 2, 4, 6, 8] -- every 2nd element
lst[::-1]         # [9, 8, ..., 0]  -- reversed
lst[1:7:2]        # [1, 3, 5]       -- start:stop:step

# Works on strings too:
"hello"[::-1]     # "olleh"

# Assignment with slices:
lst[2:5] = [20, 30, 40]
```

In ML code, you'll see slicing on tensors constantly:

```python
# Get first 10 samples from a batch:
batch = data[:10]

# Get all rows, last column:
last_col = tensor[:, -1]

# Get every other row:
subset = tensor[::2]
```

### Unpacking (like destructuring)

**TypeScript:**
```typescript
const [a, b, c] = [1, 2, 3];
const [first, ...rest] = [1, 2, 3, 4];
const { name, age } = person;
```

**Python:**
```python
a, b, c = [1, 2, 3]
first, *rest = [1, 2, 3, 4]         # rest = [2, 3, 4]
*init, last = [1, 2, 3, 4]          # init = [1, 2, 3], last = 4
a, *_, b = [1, 2, 3, 4, 5]          # a=1, b=5, _ is discarded

# No direct dict destructuring, but you can do:
name, age = person["name"], person["age"]

# Swap values (no temp variable):
a, b = b, a

# Unpacking function returns:
loss, accuracy = evaluate(model)

# Ignore values with _:
_, accuracy = evaluate(model)
```

### Walrus Operator (:=)

The walrus operator assigns a value as part of an expression. Like doing assignment
inside an `if` condition.

**TypeScript (you'd use a separate line):**
```typescript
const match = text.match(pattern);
if (match) {
    process(match);
}
```

**Python:**
```python
# Without walrus:
match = re.search(pattern, text)
if match:
    process(match)

# With walrus:
if match := re.search(pattern, text):
    process(match)

# Useful in while loops:
while chunk := f.read(8192):
    process(chunk)

# In comprehensions:
results = [y for x in data if (y := expensive(x)) > threshold]
```

You won't use it constantly, but it is nice to recognize when you see it.

### Generator Functions (yield)

**TypeScript:**
```typescript
function* range(n: number): Generator<number> {
    for (let i = 0; i < n; i++) {
        yield i;
    }
}
for (const i of range(10)) { /* ... */ }
```

**Python:**
```python
def data_batches(data, batch_size: int):
    """Yield successive batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

for batch in data_batches(training_data, 32):
    train_step(batch)
```

Generators are lazy -- they compute values on demand. Very common in ML for data loading.

Generator expressions (like lazy list comprehensions):

```python
# List comprehension -- builds entire list in memory:
squares = [x ** 2 for x in range(1_000_000)]

# Generator expression -- computes lazily:
squares = (x ** 2 for x in range(1_000_000))

# Use generators when you don't need all values at once:
total = sum(x ** 2 for x in range(1_000_000))    # memory-efficient
```

---

## 13. Package Management

### The Ecosystem

| Node/TS | Python | Notes |
|---|---|---|
| `package.json` | `pyproject.toml` or `requirements.txt` | `pyproject.toml` is the modern standard |
| `node_modules/` | `.venv/` | virtual environment |
| `npm` / `yarn` / `pnpm` | `pip` / `uv` / `poetry` / `conda` | `pip` is standard; `uv` is fast and gaining adoption |
| `npx` | `python -m` | run a module as a script |
| `package-lock.json` | `requirements.txt` (pinned) | `pip freeze > requirements.txt` |
| `.npmrc` | `pip.conf` | rarely needed |
| `nvm` | `pyenv` | manage Python versions |

### requirements.txt

The simplest way to pin dependencies. Like a very basic `package.json` dependencies list:

```
# requirements.txt
torch>=2.0.0
numpy>=1.24.0
transformers>=4.30.0
datasets
tqdm
```

```bash
pip install -r requirements.txt
pip freeze > requirements.txt    # snapshot current versions
```

### pyproject.toml (like package.json)

The modern standard for Python project configuration:

```toml
[project]
name = "my-transformer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "transformers>=4.30",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff",
    "mypy",
]

[project.scripts]
train = "my_transformer.cli:main"
```

```bash
pip install .                 # install the project
pip install -e .              # install in editable/dev mode (like npm link)
pip install -e ".[dev]"       # install with dev dependencies
```

### uv (the fast alternative)

`uv` is a modern Python package manager (like pnpm for Node). It is significantly faster
than pip:

```bash
uv venv                     # create venv
uv pip install torch         # install package
uv pip install -r requirements.txt
uv pip compile requirements.in -o requirements.txt  # lock dependencies
```

### conda (for ML specifically)

You'll encounter conda in ML because it manages non-Python dependencies (CUDA, cuDNN)
that pip cannot. If you need GPU-accelerated PyTorch:

```bash
conda create -n myenv python=3.11
conda activate myenv
conda install pytorch torchvision -c pytorch
```

---

## 14. Testing

### pytest (like Jest/Vitest)

**TypeScript (Jest):**
```typescript
describe("tokenizer", () => {
    test("splits words", () => {
        expect(tokenize("hello world")).toEqual(["hello", "world"]);
    });

    test("handles empty string", () => {
        expect(tokenize("")).toEqual([]);
    });
});
```

**Python (pytest):**
```python
# test_tokenizer.py

def test_splits_words():
    assert tokenize("hello world") == ["hello", "world"]

def test_handles_empty_string():
    assert tokenize("") == []
```

Key differences:
- No `describe` blocks. Just functions prefixed with `test_`.
- No `expect().toBe()` chains. Just use `assert`.
- File names must start with `test_` or end with `_test.py`.
- Run with `pytest` from the command line.

### Fixtures (like beforeEach/setup)

```python
import pytest

@pytest.fixture
def model():
    """Create a fresh model for each test that requests it."""
    return Transformer(d_model=64, n_heads=4, n_layers=2)

@pytest.fixture
def sample_batch():
    return torch.randn(2, 10, 64)

def test_forward_shape(model, sample_batch):
    """Fixtures are injected by name as function arguments."""
    output = model(sample_batch)
    assert output.shape == sample_batch.shape

def test_output_not_nan(model, sample_batch):
    output = model(sample_batch)
    assert not torch.isnan(output).any()
```

### Parametrize (like test.each)

```python
@pytest.mark.parametrize("input,expected", [
    ("hello world", ["hello", "world"]),
    ("", []),
    ("single", ["single"]),
])
def test_tokenize(input, expected):
    assert tokenize(input) == expected
```

### Running Tests

```bash
pytest                            # run all tests
pytest tests/test_model.py        # run one file
pytest -k "test_forward"          # run tests matching a pattern
pytest -x                         # stop on first failure
pytest -v                         # verbose output
pytest --tb=short                 # shorter tracebacks
```

### Approximate Comparisons (important for ML)

```python
# Floating point comparison:
assert result == pytest.approx(expected, abs=1e-6)

# For tensors, use torch's built-in:
torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
```

---

## 15. Common Gotchas

### Mutable Default Arguments

This is the most infamous Python gotcha. Default mutable arguments are shared across
all calls:

```python
# BUG: the default list is shared across all calls!
def add_item(item, items=[]):
    items.append(item)
    return items

add_item(1)    # [1]
add_item(2)    # [1, 2]  -- NOT [2]!

# Fix: use None as default, create a new list inside:
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

This applies to lists, dicts, sets -- any mutable object. You will encounter this pattern
in nearly every Python codebase. When you see `= None` as a default followed by an
`if x is None: x = []` pattern, now you know why.

### `is` vs `==`

```python
# == checks equality (like === in JS for value comparison)
# is checks identity (same object in memory)

a = [1, 2, 3]
b = [1, 2, 3]
a == b        # True  -- same contents
a is b        # False -- different objects

# Use `is` ONLY for None, True, False:
if x is None:       # correct
if x == None:       # works but bad style
if x is not None:   # correct
```

### Indentation Sensitivity

Mixed tabs and spaces will ruin your day. Configure your editor to use 4 spaces (which
it almost certainly does by default). If you get `IndentationError` or
`TabError`, check for mixed whitespace.

### Scope

**TypeScript:**
```typescript
let x = 10;
if (true) {
    let y = 20;  // block-scoped
}
// y is not accessible here
```

**Python:**
```python
x = 10
if True:
    y = 20       # NOT block-scoped! y leaks out.

print(y)         # 20 -- this works!

# Python only has function scope and global scope for variables.
# if/for/while/try blocks do NOT create a new scope.

# But list comprehension variables ARE scoped (Python 3):
result = [x for x in range(10)]
# x from the comprehension does NOT leak into the outer scope
```

This means loop variables persist after the loop:

```python
for i in range(10):
    pass
print(i)    # 9 -- still accessible
```

### Integer Division

```python
# In Python 3, / always returns a float:
10 / 3       # 3.3333...

# Use // for integer (floor) division:
10 // 3      # 3
-10 // 3     # -4 (floors toward negative infinity, not toward zero!)

# For truncation toward zero (like JS):
int(10 / 3)  # 3
int(-10 / 3) # -3
```

### Everything is an Object

```python
# Functions are objects:
def greet(name):
    return f"Hello, {name}"

my_func = greet          # no parentheses = reference, not call
my_func("Alice")         # "Hello, Alice"
funcs = [greet, str.upper, len]

# Classes are objects:
model_cls = Transformer
model = model_cls(d_model=512)
```

### Copy vs Reference

Like JS, assignment of collections creates a reference, not a copy:

```python
a = [1, 2, 3]
b = a               # b points to the same list
b.append(4)
print(a)            # [1, 2, 3, 4]  -- a is also modified!

# Shallow copy:
b = a.copy()         # or: b = list(a), b = a[:]

# Deep copy:
import copy
b = copy.deepcopy(a)
```

### String Immutability & Other Immutable Types

```python
# Strings, ints, floats, tuples, frozensets are immutable.
s = "hello"
s[0] = "H"         # TypeError!
s = "H" + s[1:]    # create a new string instead
```

### The `global` and `nonlocal` Keywords

If you need to modify an outer variable from inside a function (unusual but it comes up):

```python
count = 0

def increment():
    global count       # without this, Python would create a local `count`
    count += 1

def make_counter():
    count = 0
    def increment():
        nonlocal count  # refers to the enclosing function's `count`
        count += 1
        return count
    return increment
```

### Truthiness Traps

```python
# Watch out with 0 and empty containers:
value = 0
if value:             # False! 0 is falsy
    print("truthy")

# If 0 is a valid value, check for None explicitly:
if value is not None:
    print("has a value")

# Same issue with empty lists:
items = []
if items:             # False! empty list is falsy
    print("has items")
```

### No Switch Statement (until 3.10)

Python 3.10 introduced `match` (structural pattern matching), which is more powerful than
switch:

```python
# Python 3.10+
match command:
    case "train":
        run_training()
    case "eval" | "evaluate":
        run_evaluation()
    case _:
        print(f"Unknown command: {command}")

# Before 3.10, use if/elif:
if command == "train":
    run_training()
elif command in ("eval", "evaluate"):
    run_evaluation()
else:
    print(f"Unknown command: {command}")
```

---

## Quick Reference Card

| TypeScript | Python |
|---|---|
| `console.log()` | `print()` |
| `typeof x` | `type(x)` |
| `x instanceof Foo` | `isinstance(x, Foo)` |
| `Object.keys(d)` | `d.keys()` or `list(d.keys())` |
| `Object.entries(d)` | `d.items()` |
| `arr.length` | `len(arr)` |
| `arr.push(x)` | `arr.append(x)` |
| `arr.pop()` | `arr.pop()` |
| `arr.includes(x)` | `x in arr` |
| `arr.map(fn)` | `[fn(x) for x in arr]` |
| `arr.filter(fn)` | `[x for x in arr if fn(x)]` |
| `arr.reduce(fn, init)` | `functools.reduce(fn, arr, init)` |
| `arr.find(fn)` | `next((x for x in arr if fn(x)), None)` |
| `arr.flat()` | `[x for sub in arr for x in sub]` |
| `arr.slice(a, b)` | `arr[a:b]` |
| `arr.splice(i, 1)` | `del arr[i]` or `arr.pop(i)` |
| `[...a, ...b]` | `[*a, *b]` |
| `{...a, ...b}` | `{**a, **b}` |
| `str.startsWith()` | `str.startswith()` |
| `str.includes(s)` | `s in str` |
| `str.split(" ")` | `str.split(" ")` |
| `str.trim()` | `str.strip()` |
| `str.replace()` | `str.replace()` |
| `str.toUpperCase()` | `str.upper()` |
| `Math.floor()` | `math.floor()` or `//` |
| `Math.max(a, b)` | `max(a, b)` |
| `Math.random()` | `random.random()` |
| `JSON.parse()` | `json.loads()` |
| `JSON.stringify()` | `json.dumps()` |
| `Promise` / `async/await` | `asyncio` / `async/await` |
| `setTimeout()` | `asyncio.sleep()` or `time.sleep()` |
| `null` / `undefined` | `None` |
| `true` / `false` | `True` / `False` |
| `&&` / `\|\|` / `!` | `and` / `or` / `not` |
| `===` / `!==` | `==` / `!=` |
| `condition ? a : b` | `a if condition else b` |

---

## What's Next

You now have enough Python to read and write ML code. The next doc covers NumPy and
PyTorch tensors, which is where the actual model building happens. The syntax above is
just the container -- tensors are where the work lives.
