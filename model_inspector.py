"""Static model structure inspector for SGLang model files.

Parses model Python source files using the `ast` module (no GPU, no model loading)
and produces a readable nn.Module hierarchy tree + config metadata summary.

Usage:
    python model_inspector.py <model_file.py> [options]

Examples:
    python model_inspector.py /path/to/deepseek_v2.py --list-classes
    python model_inspector.py /path/to/deepseek_v2.py --root DeepseekV2ForCausalLM
    python model_inspector.py /path/to/deepseek_v2.py --config /path/to/config.json -o out.txt
"""

import argparse
import ast
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModuleAssignment:
    """Represents a `self.xxx = SomeModule(...)` found in __init__."""

    attr_name: str
    class_name: str
    raw_args: str
    line_no: int
    is_conditional: bool = False
    condition_text: str = ""


@dataclass
class ClassInfo:
    """Parsed class with its name, bases, module assignments, and config accesses."""

    name: str
    bases: List[str]
    assignments: List[ModuleAssignment] = field(default_factory=list)
    config_accesses: List[str] = field(default_factory=list)
    line_no: int = 0


@dataclass
class ModuleTree:
    """Tree node for the module hierarchy."""

    class_name: str
    attr_name: str
    children: List["ModuleTree"] = field(default_factory=list)
    multiplicity: str = ""
    is_conditional: bool = False
    condition_text: str = ""
    line_no: int = 0


@dataclass
class ConfigMetadata:
    """Wrapper around relevant config.json key-value pairs."""

    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Known module heuristics
# ---------------------------------------------------------------------------

_MODULE_KEYWORDS = {
    "Linear", "Norm", "Embedding", "Attention", "MoE", "MLP",
    "Processor", "TopK", "Head", "Layer", "Conv", "Pool",
    "Dropout", "BatchNorm", "LayerNorm", "RMSNorm", "GroupNorm",
    "Module", "Sequential", "ModuleList", "Indexer", "Gate",
    "Dispatcher", "Communicator", "Rotary", "Rope",
}

_NON_MODULE_NAMES = {
    "Parameter", "Tensor", "bool", "int", "float", "str", "list", "dict",
    "tuple", "set", "None", "True", "False", "torch", "F", "nn",
    "SiluAndMul", "Stage", "Enum", "LazyValue", "PackWeightMethod",
}

_NON_MODULE_PREFIXES = {"nn.Parameter", "torch."}


def _looks_like_module(class_name: str, known_classes: set) -> bool:
    """Heuristic: does this class name look like an nn.Module subclass?"""
    # Factory calls like get_moe_impl_class()() are likely module constructors
    if class_name.endswith("()"):
        inner = class_name[:-2]
        # Heuristic: factory functions with "moe", "impl", "class" in name
        lower_inner = inner.lower()
        if any(kw in lower_inner for kw in ["moe", "impl", "class", "layer", "module"]):
            return True
    if class_name in _NON_MODULE_NAMES:
        return False
    for prefix in _NON_MODULE_PREFIXES:
        if class_name.startswith(prefix) and class_name != "nn.Module":
            # nn.Parameter, torch.zeros, etc.
            if "Linear" not in class_name and "Norm" not in class_name:
                return False
    # Filter out static/class method calls like Foo.bar_method()
    if "." in class_name and not class_name.startswith("nn."):
        parts = class_name.rsplit(".", 1)
        method = parts[-1]
        # If the last part starts lowercase, it's a method call, not a constructor
        if method and method[0].islower():
            return False
    if class_name in known_classes:
        return True
    for keyword in _MODULE_KEYWORDS:
        if keyword in class_name:
            return True
    # PascalCase heuristic: at least two uppercase letters and no dots (except nn.)
    if class_name.startswith("nn."):
        return True
    upper_count = sum(1 for c in class_name if c.isupper())
    if upper_count >= 2 and "." not in class_name:
        return True
    return False


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _get_call_class_name(node: ast.Call) -> Optional[str]:
    """Extract class name from a Call node.

    Handles:
      - ClassName(...)           -> "ClassName"
      - module.ClassName(...)    -> "module.ClassName"
      - factory()(...)           -> inner call name + "()"
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        value = func.value
        if isinstance(value, ast.Name):
            return f"{value.id}.{func.attr}"
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            return f"{value.value.id}.{value.attr}.{func.attr}"
    # Factory call: get_moe_impl_class(quant_config)(...)
    if isinstance(func, ast.Call):
        inner = _get_call_class_name(func)
        if inner:
            return f"{inner}()"
    return None


def _get_source_segment(source_lines: List[str], node: ast.AST) -> str:
    """Get a simplified source representation of an AST node."""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _extract_lambda_body_class(node: ast.Call) -> Optional[str]:
    """From make_layers(..., lambda idx, prefix: ClassName(...), ...),
    extract the ClassName from the lambda body."""
    for arg in list(node.args) + [kw.value for kw in node.keywords]:
        if isinstance(arg, ast.Lambda):
            body = arg.body
            if isinstance(body, ast.Call):
                return _get_call_class_name(body)
    return None


# ---------------------------------------------------------------------------
# ModelFileParser
# ---------------------------------------------------------------------------

class ModelFileParser:
    """AST-based parser for model Python source files."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, "r") as f:
            self.source = f.read()
        self.source_lines = self.source.splitlines()
        self.tree = ast.parse(self.source, filename=filepath)
        self.classes: Dict[str, ClassInfo] = {}

    def parse(self) -> Dict[str, ClassInfo]:
        """Parse all classes in the file."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                bases = self._extract_bases(node)
                info = ClassInfo(
                    name=node.name,
                    bases=bases,
                    line_no=node.lineno,
                )
                self.classes[node.name] = info

        # Second pass: extract __init__ assignments now that we know all class names
        known_classes = set(self.classes.keys())
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name in self.classes:
                info = self.classes[node.name]
                init_method = self._find_init(node)
                if init_method:
                    info.assignments = self._extract_init_assignments(
                        init_method.body, known_classes
                    )
                    info.config_accesses = self._extract_config_accesses(init_method)

        return self.classes

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))
        return bases

    def _find_init(self, class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
        """Find the __init__ method in a class."""
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return item
        return None

    def _extract_init_assignments(
        self,
        body: List[ast.stmt],
        known_classes: set,
        is_conditional: bool = False,
        condition_text: str = "",
    ) -> List[ModuleAssignment]:
        """Walk __init__ body (including if/else) to find self.xxx = ClassName(...)."""
        assignments = []

        for stmt in body:
            # self.xxx = ClassName(...)
            if isinstance(stmt, ast.Assign):
                assignments.extend(
                    self._process_assign(
                        stmt, known_classes, is_conditional, condition_text
                    )
                )

            # if condition: ... else: ...
            elif isinstance(stmt, ast.If):
                cond_text = ast.unparse(stmt.test)
                assignments.extend(
                    self._extract_init_assignments(
                        stmt.body, known_classes,
                        is_conditional=True,
                        condition_text=cond_text,
                    )
                )
                if stmt.orelse:
                    else_text = f"not ({cond_text})"
                    # If orelse is another If (elif), keep its own condition
                    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
                        assignments.extend(
                            self._extract_init_assignments(
                                stmt.orelse, known_classes,
                                is_conditional=True,
                                condition_text="",
                            )
                        )
                    else:
                        assignments.extend(
                            self._extract_init_assignments(
                                stmt.orelse, known_classes,
                                is_conditional=True,
                                condition_text=else_text,
                            )
                        )

        return assignments

    def _process_assign(
        self,
        stmt: ast.Assign,
        known_classes: set,
        is_conditional: bool,
        condition_text: str,
    ) -> List[ModuleAssignment]:
        """Process a single assignment statement."""
        results = []

        # Handle tuple unpacking: self.layers, self.start_layer, self.end_layer = make_layers(...)
        if (
            len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Tuple)
            and isinstance(stmt.value, ast.Call)
        ):
            target_tuple = stmt.targets[0]
            call_name = _get_call_class_name(stmt.value)
            if call_name == "make_layers":
                # Find the lambda body to get the constructed class
                constructed_class = _extract_lambda_body_class(stmt.value)
                if constructed_class:
                    # Find the first self.xxx in the tuple (usually self.layers)
                    for elt in target_tuple.elts:
                        if (
                            isinstance(elt, ast.Attribute)
                            and isinstance(elt.value, ast.Name)
                            and elt.value.id == "self"
                        ):
                            attr_name = elt.attr
                            # Get num_layers arg
                            num_layers_arg = ""
                            if stmt.value.args:
                                num_layers_arg = ast.unparse(stmt.value.args[0])
                            results.append(ModuleAssignment(
                                attr_name=attr_name,
                                class_name=constructed_class,
                                raw_args=f"make_layers({num_layers_arg}, ...)",
                                line_no=stmt.lineno,
                                is_conditional=is_conditional,
                                condition_text=condition_text,
                            ))
                            break
            return results

        # Standard: self.xxx = ClassName(...)
        for target in stmt.targets:
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                continue

            attr_name = target.attr
            value = stmt.value

            if not isinstance(value, ast.Call):
                continue

            class_name = _get_call_class_name(value)
            if not class_name:
                continue

            if not _looks_like_module(class_name, known_classes):
                continue

            raw_args = ast.unparse(value)
            results.append(ModuleAssignment(
                attr_name=attr_name,
                class_name=class_name,
                raw_args=raw_args,
                line_no=stmt.lineno,
                is_conditional=is_conditional,
                condition_text=condition_text,
            ))

        return results

    def _extract_config_accesses(self, func_node: ast.FunctionDef) -> List[str]:
        """Find config.xxx attribute accesses in the function."""
        accesses = set()
        for node in ast.walk(func_node):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "config"
            ):
                accesses.add(node.attr)
        return sorted(accesses)


# ---------------------------------------------------------------------------
# HierarchyBuilder
# ---------------------------------------------------------------------------

class HierarchyBuilder:
    """Builds a module tree from parsed class info."""

    def __init__(self, class_map: Dict[str, ClassInfo]):
        self.class_map = class_map

    def build(self, root_class_name: str) -> ModuleTree:
        """Build the module hierarchy starting from root_class_name."""
        if root_class_name not in self.class_map:
            return ModuleTree(class_name=root_class_name, attr_name="(root)")
        return self._build_node(root_class_name, "(root)", visited=set())

    def _build_node(
        self, class_name: str, attr_name: str, visited: set
    ) -> ModuleTree:
        node = ModuleTree(class_name=class_name, attr_name=attr_name)

        if class_name in visited:
            # Avoid infinite recursion
            return node
        visited = visited | {class_name}

        info = self.class_map.get(class_name)
        if not info:
            return node

        node.line_no = info.line_no

        for assignment in info.assignments:
            child_class = assignment.class_name
            # Strip factory call markers
            resolved_class = child_class.rstrip("()")

            multiplicity = self._detect_multiplicity(assignment)

            if resolved_class in self.class_map:
                child_node = self._build_node(resolved_class, assignment.attr_name, visited)
            else:
                child_node = ModuleTree(
                    class_name=child_class,
                    attr_name=assignment.attr_name,
                )

            child_node.multiplicity = multiplicity
            child_node.is_conditional = assignment.is_conditional
            child_node.condition_text = assignment.condition_text
            child_node.line_no = assignment.line_no
            node.children.append(child_node)

        return node

    def _detect_multiplicity(self, assignment: ModuleAssignment) -> str:
        """Detect if this assignment creates multiple layers."""
        if "make_layers" in assignment.raw_args:
            # Try to extract the count argument
            # Pattern: make_layers(config.num_hidden_layers, ...)
            raw = assignment.raw_args
            if "make_layers(" in raw:
                inner = raw.split("make_layers(", 1)[1]
                count_arg = inner.split(",", 1)[0].strip()
                return f"x N ({count_arg})"
        if "ModuleList" in assignment.raw_args:
            return "x N"
        return ""


# ---------------------------------------------------------------------------
# ConfigParser
# ---------------------------------------------------------------------------

class ConfigParser:
    """Parse HuggingFace config.json and extract relevant keys."""

    RELEVANT_KEYS = [
        "architectures", "model_type",
        "hidden_size", "intermediate_size", "moe_intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "n_group", "topk_group", "topk_method",
        "first_k_dense_replace", "moe_layer_freq",
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "q_lora_rank", "kv_lora_rank",
        "vocab_size", "max_position_embeddings",
        "rms_norm_eps", "rope_theta", "rope_scaling",
        "routed_scaling_factor",
        "tie_word_embeddings",
    ]

    @classmethod
    def parse(cls, config_path: str) -> ConfigMetadata:
        """Read config.json and extract relevant fields."""
        with open(config_path, "r") as f:
            raw = json.load(f)

        data = {}
        for key in cls.RELEVANT_KEYS:
            if key in raw:
                data[key] = raw[key]

        return ConfigMetadata(data=data)


# ---------------------------------------------------------------------------
# OutputFormatter
# ---------------------------------------------------------------------------

class OutputFormatter:
    """Format and print the module hierarchy and config."""

    def __init__(self, show_line_numbers: bool = False):
        self.show_line_numbers = show_line_numbers

    def format_tree(self, tree: ModuleTree) -> str:
        """Format the module tree as an indented string."""
        lines = []
        lines.append(f"=== Module Hierarchy: {tree.class_name} ===")
        lines.append(tree.class_name)
        self._format_children(tree, lines, prefix="", is_last=True)
        return "\n".join(lines)

    def _format_children(
        self, node: ModuleTree, lines: List[str], prefix: str, is_last: bool
    ):
        children = node.children
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            connector = "|- "
            extension = "|   " if not is_child_last else "    "

            label = f"{child.attr_name}: {child.class_name}"
            if child.multiplicity:
                label += f" [{child.multiplicity}]"
            if child.is_conditional and child.condition_text:
                cond_display = child.condition_text
                # Truncate long conditions
                if len(cond_display) > 60:
                    cond_display = cond_display[:57] + "..."
                label += f" (conditional: {cond_display})"

            if self.show_line_numbers and child.line_no:
                label += f"  [L{child.line_no}]"

            lines.append(f"{prefix}{connector}{label}")

            if child.children:
                self._format_children(
                    child, lines, prefix=prefix + extension, is_last=is_child_last
                )

    def format_config(self, config: ConfigMetadata) -> str:
        """Format config metadata as aligned key=value table."""
        if not config.data:
            return ""
        lines = ["\n=== Config Metadata ==="]
        max_key_len = max(len(k) for k in config.data)
        for key, value in config.data.items():
            val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            lines.append(f"  {key:<{max_key_len}} = {val_str}")
        return "\n".join(lines)

    def format_profiler_tree(
        self,
        tree: ModuleTree,
        config: Optional[ConfigMetadata] = None,
        max_depth: int = 2,
    ) -> str:
        """Format as a PyTorch-profiler-style expanded tree with instance indices.

        Expands repeated layers (make_layers) into individual instances with _0, _1, ...
        and resolves conditional branches per-layer using config metadata.
        """
        lines = []
        lines.append(f"=== Module Tree (profiler-style): {tree.class_name} ===")
        lines.append(tree.class_name)

        config_data = config.data if config else {}
        self._profiler_children(tree, lines, prefix="", depth=0, max_depth=max_depth,
                                config_data=config_data, layer_id=None)
        return "\n".join(lines)

    def _profiler_children(
        self,
        node: ModuleTree,
        lines: List[str],
        prefix: str,
        depth: int,
        max_depth: int,
        config_data: Dict[str, Any],
        layer_id: Optional[int],
    ):
        """Recursively format children in profiler style."""
        children = node.children
        # Collect expanded children list (expanding repeated layers)
        expanded: List[Tuple[ModuleTree, Optional[int], bool]] = []  # (node, layer_id, is_expanded)

        for child in children:
            if child.multiplicity and "x N" in child.multiplicity:
                # Expand into individual instances
                num_layers = self._resolve_num_layers(child.multiplicity, config_data)
                if num_layers is not None:
                    for idx in range(num_layers):
                        expanded.append((child, idx, True))
                else:
                    expanded.append((child, None, False))
            else:
                # Skip conditional variants — resolve which one applies
                expanded.append((child, layer_id, False))

        # De-duplicate conditionals: group by attr_name, pick the right branch
        expanded = self._resolve_conditionals(expanded, config_data)

        for i, (child, lid, is_expanded) in enumerate(expanded):
            is_last = (i == len(expanded) - 1)
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            extension = "    " if is_last else "\u2502   "

            # Build label
            class_name = child.class_name
            if is_expanded and lid is not None:
                label = f"{class_name}_{lid}"
            else:
                label = f"{class_name}_0"

            if self.show_line_numbers and child.line_no:
                label += f"  [L{child.line_no}]"

            lines.append(f"{prefix}{connector}{label}")

            # Recurse if under depth limit and has children
            if depth + 1 < max_depth and child.children:
                self._profiler_children(
                    child, lines, prefix=prefix + extension,
                    depth=depth + 1, max_depth=max_depth,
                    config_data=config_data,
                    layer_id=lid if lid is not None else layer_id,
                )

    def _resolve_num_layers(self, multiplicity: str, config_data: Dict[str, Any]) -> Optional[int]:
        """Resolve 'x N (config.num_hidden_layers)' to an actual integer."""
        # Try to find the config key in the multiplicity string
        if "num_hidden_layers" in multiplicity:
            val = config_data.get("num_hidden_layers")
            if isinstance(val, int):
                return val
        if "num_layers" in multiplicity:
            val = config_data.get("num_layers")
            if isinstance(val, int):
                return val
        # Try generic extraction
        import re
        m = re.search(r'config\.(\w+)', multiplicity)
        if m:
            val = config_data.get(m.group(1))
            if isinstance(val, int):
                return val
        return None

    def _resolve_conditionals(
        self,
        items: List[Tuple[ModuleTree, Optional[int], bool]],
        config_data: Dict[str, Any],
    ) -> List[Tuple[ModuleTree, Optional[int], bool]]:
        """For groups sharing the same attr_name, pick the right conditional branch.

        Uses config_data to resolve conditions like `self.is_layer_sparse` based on
        first_k_dense_replace and moe_layer_freq.
        """
        result = []
        i = 0
        while i < len(items):
            child, lid, is_exp = items[i]
            if not child.is_conditional:
                result.append((child, lid, is_exp))
                i += 1
                continue

            # Collect all conditionals with the same attr_name at this position
            group = [(child, lid, is_exp)]
            j = i + 1
            while j < len(items):
                next_child, next_lid, next_exp = items[j]
                if next_child.attr_name == child.attr_name and next_child.is_conditional:
                    group.append((next_child, next_lid, next_exp))
                    j += 1
                else:
                    break

            # Try to resolve which branch applies
            picked = self._pick_conditional(group, config_data)
            result.append(picked)
            i = j

        return result

    def _pick_conditional(
        self,
        group: List[Tuple[ModuleTree, Optional[int], bool]],
        config_data: Dict[str, Any],
    ) -> Tuple[ModuleTree, Optional[int], bool]:
        """Pick the correct conditional branch for a layer.

        For `self.is_layer_sparse` conditions, use first_k_dense_replace + moe_layer_freq.
        """
        if len(group) == 1:
            return group[0]

        child, lid, is_exp = group[0]

        # Try to resolve layer-sparsity condition
        if "is_layer_sparse" in child.condition_text and lid is not None:
            first_k = config_data.get("first_k_dense_replace", 0)
            moe_freq = config_data.get("moe_layer_freq", 1)
            is_sparse = lid >= first_k and moe_freq > 0 and lid % moe_freq == 0
            for g_child, g_lid, g_exp in group:
                cond = g_child.condition_text
                if is_sparse and "is_layer_sparse" in cond and "not" not in cond:
                    return (g_child, g_lid, g_exp)
                if not is_sparse and ("not" in cond or "else" in cond.lower()):
                    return (g_child, g_lid, g_exp)
            # Fallback: pick based on sparse = first branch
            return group[0] if is_sparse else group[-1]

        # For pp_group conditions, default to first rank (most common view)
        if "pp_group.is_first_rank" in child.condition_text:
            for g_child, g_lid, g_exp in group:
                if "not" not in g_child.condition_text:
                    return (g_child, g_lid, g_exp)

        if "pp_group.is_last_rank" in child.condition_text:
            for g_child, g_lid, g_exp in group:
                if "not" not in g_child.condition_text:
                    return (g_child, g_lid, g_exp)

        # Default: first branch
        return group[0]

    def format_class_list(self, classes: Dict[str, ClassInfo]) -> str:
        """Format a list of all classes."""
        lines = ["=== Classes Found ==="]
        for name, info in classes.items():
            bases_str = ", ".join(info.bases) if info.bases else ""
            n_assignments = len(info.assignments)
            line = f"  L{info.line_no:<5} {name}"
            if bases_str:
                line += f"({bases_str})"
            line += f"  [{n_assignments} module assignments]"
            if info.config_accesses:
                line += f"  config: {', '.join(info.config_accesses[:5])}"
                if len(info.config_accesses) > 5:
                    line += f" (+{len(info.config_accesses) - 5} more)"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-detect root class
# ---------------------------------------------------------------------------

def _auto_detect_root(classes: Dict[str, ClassInfo]) -> Optional[str]:
    """Auto-detect the root class (typically *ForCausalLM)."""
    # Prefer classes ending with ForCausalLM that are not subclasses of other local classes
    causal_lm = [
        name for name in classes
        if name.endswith("ForCausalLM")
    ]
    if causal_lm:
        # If there are subclass relationships, pick the base one
        # (e.g. DeepseekV2ForCausalLM over DeepseekV3ForCausalLM)
        for name in causal_lm:
            info = classes[name]
            # Check if any base is NOT another local ForCausalLM
            has_local_parent = any(
                base in classes and base.endswith("ForCausalLM")
                for base in info.bases
            )
            if not has_local_parent:
                return name
        return causal_lm[0]

    # Fallback: look for *ForConditionalGeneration, *Model, etc.
    for suffix in ["ForConditionalGeneration", "Model"]:
        matches = [name for name in classes if name.endswith(suffix)]
        if matches:
            return matches[0]

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Static model structure inspector for SGLang model files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python model_inspector.py deepseek_v2.py --list-classes
  python model_inspector.py deepseek_v2.py --root DeepseekV2ForCausalLM
  python model_inspector.py deepseek_v2.py --config config.json -o out.txt
  python model_inspector.py deepseek_v2.py --show-line-numbers
  python model_inspector.py deepseek_v2.py --profiler-tree --config config.json
  python model_inspector.py deepseek_v2.py --profiler-tree --config config.json --depth 3
""",
    )
    parser.add_argument("model_file", help="Path to model Python source file")
    parser.add_argument(
        "--root", metavar="CLASS",
        help="Root class name (default: auto-detect *ForCausalLM)",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="Path to HuggingFace config.json",
    )
    parser.add_argument(
        "--list-classes", action="store_true",
        help="List all classes in the file and exit",
    )
    parser.add_argument(
        "--output", "-o", metavar="PATH",
        help="Save output to file",
    )
    parser.add_argument(
        "--show-line-numbers", action="store_true",
        help="Show source line numbers in the tree",
    )
    parser.add_argument(
        "--profiler-tree", action="store_true",
        help="PyTorch-profiler-style tree with expanded layer instances",
    )
    parser.add_argument(
        "--depth", type=int, default=2, metavar="N",
        help="Max depth for --profiler-tree (default: 2)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.model_file):
        print(f"Error: File not found: {args.model_file}", file=sys.stderr)
        sys.exit(1)

    # Parse the model file
    file_parser = ModelFileParser(args.model_file)
    classes = file_parser.parse()

    if not classes:
        print("No classes found in the file.", file=sys.stderr)
        sys.exit(1)

    formatter = OutputFormatter(show_line_numbers=args.show_line_numbers)
    output_parts = []

    # --list-classes mode
    if args.list_classes:
        output_parts.append(formatter.format_class_list(classes))
    else:
        # Determine root class
        root_class = args.root
        if not root_class:
            root_class = _auto_detect_root(classes)
            if not root_class:
                print(
                    "Error: Could not auto-detect root class. "
                    "Use --root to specify.",
                    file=sys.stderr,
                )
                sys.exit(1)

        if root_class not in classes:
            print(
                f"Error: Class '{root_class}' not found in {args.model_file}",
                file=sys.stderr,
            )
            print(f"Available classes: {', '.join(classes.keys())}", file=sys.stderr)
            sys.exit(1)

        # Build hierarchy
        builder = HierarchyBuilder(classes)
        tree = builder.build(root_class)

        if args.profiler_tree:
            # Need config for layer expansion
            config = None
            if args.config and os.path.isfile(args.config):
                config = ConfigParser.parse(args.config)
            output_parts.append(
                formatter.format_profiler_tree(tree, config=config, max_depth=args.depth)
            )
        else:
            output_parts.append(formatter.format_tree(tree))

    # Config metadata (skip in profiler-tree mode — config is used internally only)
    if args.config and not args.profiler_tree:
        if not os.path.isfile(args.config):
            print(f"Warning: Config file not found: {args.config}", file=sys.stderr)
        else:
            config = ConfigParser.parse(args.config)
            config_output = formatter.format_config(config)
            if config_output:
                output_parts.append(config_output)

    # Emit output
    full_output = "\n\n".join(output_parts) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(full_output)
        print(f"Output saved to {args.output}")
    else:
        print(full_output, end="")


if __name__ == "__main__":
    main()
