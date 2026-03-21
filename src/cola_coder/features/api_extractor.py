"""Code API Extractor.

Extract the public API from Python modules:
  - Classes with their methods and attributes
  - Module-level functions with signatures
  - Constants / module-level variables
  - Type annotations (where available)
  - Docstrings

Generates API documentation stubs and structured type info,
suitable for generating documentation or training data.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParamInfo:
    """Information about a single function parameter."""

    name: str
    annotation: str | None = None
    default: str | None = None
    kind: str = "POSITIONAL_OR_KEYWORD"  # mirrors inspect.Parameter.kind names

    def to_stub(self) -> str:
        s = self.name
        if self.annotation:
            s += f": {self.annotation}"
        if self.default is not None:
            s += f" = {self.default}"
        return s


@dataclass
class FunctionInfo:
    """Information about a function or method."""

    name: str
    params: list[ParamInfo] = field(default_factory=list)
    return_annotation: str | None = None
    docstring: str | None = None
    is_async: bool = False
    is_property: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_private: bool = False  # starts with _
    is_dunder: bool = False   # starts and ends with __

    def signature(self) -> str:
        params_str = ", ".join(p.to_stub() for p in self.params)
        prefix = "async " if self.is_async else ""
        ret = f" -> {self.return_annotation}" if self.return_annotation else ""
        return f"{prefix}def {self.name}({params_str}){ret}"

    def to_stub(self, indent: int = 0) -> str:
        pad = " " * indent
        decorators = []
        if self.is_property:
            decorators.append(f"{pad}@property")
        if self.is_classmethod:
            decorators.append(f"{pad}@classmethod")
        if self.is_staticmethod:
            decorators.append(f"{pad}@staticmethod")
        lines = decorators + [f"{pad}{self.signature()}:"]
        if self.docstring:
            wrapped = textwrap.indent(f'"""{self.docstring}"""', pad + "    ")
            lines.append(wrapped)
        else:
            lines.append(f"{pad}    ...")
        return "\n".join(lines)


@dataclass
class ClassInfo:
    """Information about a class definition."""

    name: str
    bases: list[str] = field(default_factory=list)
    docstring: str | None = None
    methods: list[FunctionInfo] = field(default_factory=list)
    class_variables: list[tuple[str, str | None]] = field(default_factory=list)
    is_dataclass: bool = False

    def public_methods(self) -> list[FunctionInfo]:
        return [m for m in self.methods if not m.is_private or m.is_dunder]

    def to_stub(self) -> str:
        bases_str = f"({', '.join(self.bases)})" if self.bases else ""
        lines = [f"class {self.name}{bases_str}:"]
        if self.docstring:
            lines.append(f'    """{self.docstring}"""')
        for var_name, var_type in self.class_variables:
            if var_type:
                lines.append(f"    {var_name}: {var_type}")
            else:
                lines.append(f"    {var_name}: ...")
        for method in self.methods:
            if not method.is_private or method.is_dunder:
                lines.append("")
                lines.append(method.to_stub(indent=4))
        if not lines[1:]:
            lines.append("    ...")
        return "\n".join(lines)


@dataclass
class ConstantInfo:
    """A module-level constant or variable."""

    name: str
    value_repr: str | None = None
    annotation: str | None = None

    def to_stub(self) -> str:
        if self.annotation:
            return f"{self.name}: {self.annotation}"
        if self.value_repr:
            return f"{self.name} = {self.value_repr}"
        return f"{self.name}: ..."


@dataclass
class ModuleAPI:
    """Complete public API extracted from a module."""

    module_name: str
    docstring: str | None = None
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    constants: list[ConstantInfo] = field(default_factory=list)

    def public_functions(self) -> list[FunctionInfo]:
        return [f for f in self.functions if not f.is_private]

    def public_classes(self) -> list[ClassInfo]:
        return [c for c in self.classes]

    def to_stub(self) -> str:
        """Generate a .pyi-style stub string for the module."""
        lines: list[str] = []
        if self.docstring:
            lines.append(f'"""{self.docstring}"""')
            lines.append("")
        for const in self.constants:
            lines.append(const.to_stub())
        if self.constants:
            lines.append("")
        for func in self.public_functions():
            lines.append(func.to_stub())
            lines.append("")
        for cls in self.classes:
            lines.append(cls.to_stub())
            lines.append("")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate Markdown documentation for the API."""
        lines = [f"# {self.module_name}"]
        if self.docstring:
            lines.append("")
            lines.append(self.docstring)
        if self.classes:
            lines.append("")
            lines.append("## Classes")
            for cls in self.classes:
                lines.append(f"\n### `{cls.name}`")
                if cls.docstring:
                    lines.append(f"\n{cls.docstring}")
                if cls.public_methods():
                    lines.append("\n**Methods:**")
                    for method in cls.public_methods():
                        lines.append(f"- `{method.signature()}`")
        if self.public_functions():
            lines.append("")
            lines.append("## Functions")
            for func in self.public_functions():
                lines.append(f"\n### `{func.signature()}`")
                if func.docstring:
                    lines.append(f"\n{func.docstring}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class APIExtractor:
    """Extract public API from Python source code.

    Parameters
    ----------
    include_private:
        If True, include private members (starting with _).
    include_dunder:
        If True, include dunder methods (__init__, etc.).
    module_name:
        Name to use for the module in the report.
    """

    def __init__(
        self,
        include_private: bool = False,
        include_dunder: bool = True,
        module_name: str = "<module>",
    ) -> None:
        self.include_private = include_private
        self.include_dunder = include_dunder
        self.module_name = module_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, code: str) -> ModuleAPI:
        """Parse *code* and return a ModuleAPI."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ModuleAPI(module_name=self.module_name)

        api = ModuleAPI(
            module_name=self.module_name,
            docstring=ast.get_docstring(tree),
        )

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._extract_function(node)
                if self._should_include(func):
                    api.functions.append(func)
            elif isinstance(node, ast.ClassDef):
                cls = self._extract_class(node)
                api.classes.append(cls)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("__"):
                        const = ConstantInfo(
                            name=target.id,
                            value_repr=ast.unparse(node.value),
                        )
                        api.constants.append(const)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                const = ConstantInfo(
                    name=node.target.id,
                    annotation=ast.unparse(node.annotation),
                    value_repr=ast.unparse(node.value) if node.value else None,
                )
                api.constants.append(const)

        return api

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        decorators: list[str] | None = None,
    ) -> FunctionInfo:
        name = node.name
        is_dunder = name.startswith("__") and name.endswith("__")
        is_private = name.startswith("_") and not is_dunder

        is_property = False
        is_classmethod = False
        is_staticmethod = False
        for dec in node.decorator_list:
            dec_str = ast.unparse(dec)
            if dec_str == "property":
                is_property = True
            elif dec_str == "classmethod":
                is_classmethod = True
            elif dec_str == "staticmethod":
                is_staticmethod = True

        params = self._extract_params(node.args)
        ret = ast.unparse(node.returns) if node.returns else None

        return FunctionInfo(
            name=name,
            params=params,
            return_annotation=ret,
            docstring=ast.get_docstring(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_property=is_property,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            is_private=is_private,
            is_dunder=is_dunder,
        )

    def _extract_params(self, args: ast.arguments) -> list[ParamInfo]:
        params: list[ParamInfo] = []
        all_args = args.posonlyargs + args.args
        defaults_offset = len(all_args) - len(args.defaults)

        for i, arg in enumerate(all_args):
            annotation = ast.unparse(arg.annotation) if arg.annotation else None
            default_idx = i - defaults_offset
            default = ast.unparse(args.defaults[default_idx]) if default_idx >= 0 else None
            kind = "POSITIONAL_ONLY" if arg in args.posonlyargs else "POSITIONAL_OR_KEYWORD"
            params.append(ParamInfo(name=arg.arg, annotation=annotation, default=default, kind=kind))

        if args.vararg:
            a = args.vararg
            params.append(ParamInfo(
                name=f"*{a.arg}",
                annotation=ast.unparse(a.annotation) if a.annotation else None,
                kind="VAR_POSITIONAL",
            ))
        for i, arg in enumerate(args.kwonlyargs):
            annotation = ast.unparse(arg.annotation) if arg.annotation else None
            kw_defaults = args.kw_defaults
            default = ast.unparse(kw_defaults[i]) if i < len(kw_defaults) and kw_defaults[i] else None
            params.append(ParamInfo(name=arg.arg, annotation=annotation, default=default, kind="KEYWORD_ONLY"))
        if args.kwarg:
            a = args.kwarg
            params.append(ParamInfo(
                name=f"**{a.arg}",
                annotation=ast.unparse(a.annotation) if a.annotation else None,
                kind="VAR_KEYWORD",
            ))
        return params

    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        bases = [ast.unparse(b) for b in node.bases]
        is_dc = any(
            ast.unparse(d) in ("dataclass", "dataclasses.dataclass")
            for d in node.decorator_list
        )
        cls = ClassInfo(
            name=node.name,
            bases=bases,
            docstring=ast.get_docstring(node),
            is_dataclass=is_dc,
        )

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._extract_function(item)
                if self.include_dunder or not func.is_dunder:
                    if self.include_private or not func.is_private or func.is_dunder:
                        cls.methods.append(func)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                cls.class_variables.append((
                    item.target.id,
                    ast.unparse(item.annotation) if item.annotation else None,
                ))

        return cls

    def _should_include(self, func: FunctionInfo) -> bool:
        if func.is_dunder:
            return self.include_dunder
        if func.is_private:
            return self.include_private
        return True
