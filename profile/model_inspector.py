"""Static model structure inspector for SGLang model files.

Parses model Python source files using the `ast` module (no GPU, no model loading)
and produces a readable nn.Module hierarchy tree + config metadata summary.

Usage:
    python model_inspector.py <model_file.py> [options]

Examples:
    python model_inspector.py /path/to/deepseek_v2.py --list-classes
    python model_inspector.py /path/to/deepseek_v2.py --root DeepseekV2ForCausalLM
    python model_inspector.py /path/to/deepseek_v2.py --config /path/to/config.json -o out.txt

    python3 /home/yichiche/agent-box/profile/model_inspector.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py --config /data/deepseek-ai/DeepSeek-R1-0528/config.json
    python3 /home/yichiche/agent-box/profile/model_inspector.py /sgl-workspace/sglang/python/sglang/srt/models/grok.py --config /data/huggingface/hub/xai-org/grok-2/config.json
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
    # For hybrid dispatch: maps a key (e.g. "attention") to a class name
    dispatch_map: Optional[Dict[str, str]] = None


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
    # For hybrid dispatch: maps a key (e.g. "attention") to class name
    dispatch_map: Optional[Dict[str, str]] = None


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


def _extract_func_ref_name(node: ast.Call) -> Optional[str]:
    """From make_layers(N, get_layer, ...), extract the function reference name 'get_layer'."""
    if len(node.args) >= 2:
        arg = node.args[1]
        if isinstance(arg, ast.Name):
            return arg.id
    return None


def _find_local_func(body: List[ast.stmt], func_name: str) -> Optional[ast.FunctionDef]:
    """Find a local function definition in a list of statements."""
    for stmt in body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == func_name:
            return stmt
    return None


def _extract_classes_from_func(func_node: ast.FunctionDef) -> List[str]:
    """Extract class names constructed in return statements of a local function."""
    classes = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            name = _get_call_class_name(node.value)
            if name:
                classes.append(name)
    return classes


def _extract_dispatch_dict(tree: ast.Module, dict_name: str) -> Dict[str, str]:
    """Extract a module-level dict mapping string keys to class names.

    Handles patterns like:
        ALL_DECODER_LAYER_TYPES = {
            "attention": Qwen3HybridAttentionDecoderLayer,
            "linear_attention": Qwen3HybridLinearDecoderLayer,
        }
    """
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    if isinstance(node.value, ast.Dict):
                        for key, val in zip(node.value.keys, node.value.values):
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                if isinstance(val, ast.Name):
                                    result[key.value] = val.id
                                elif isinstance(val, ast.Attribute):
                                    result[key.value] = ast.unparse(val)
    return result


def _detect_dispatch_in_func(
    func_node: ast.FunctionDef, module_tree: ast.Module
) -> Optional[Dict[str, str]]:
    """Detect if a local function dispatches via a module-level dict.

    Looks for patterns like:
        layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]
        return layer_class(...)
    """
    # Look for: xxx = SOME_DICT[...]
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Subscript):
                subscript_val = node.value.value
                if isinstance(subscript_val, ast.Name):
                    dict_name = subscript_val.id
                    dispatch = _extract_dispatch_dict(module_tree, dict_name)
                    if dispatch:
                        return dispatch
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
                        init_method.body, known_classes, init_body=init_method.body
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
        init_body: Optional[List[ast.stmt]] = None,
    ) -> List[ModuleAssignment]:
        """Walk __init__ body (including if/else) to find self.xxx = ClassName(...)."""
        if init_body is None:
            init_body = body
        assignments = []

        for stmt in body:
            # self.xxx = ClassName(...)
            if isinstance(stmt, ast.Assign):
                assignments.extend(
                    self._process_assign(
                        stmt, known_classes, is_conditional, condition_text,
                        init_body=init_body,
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
                        init_body=init_body,
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
                                init_body=init_body,
                            )
                        )
                    else:
                        assignments.extend(
                            self._extract_init_assignments(
                                stmt.orelse, known_classes,
                                is_conditional=True,
                                condition_text=else_text,
                                init_body=init_body,
                            )
                        )

        return assignments

    def _process_assign(
        self,
        stmt: ast.Assign,
        known_classes: set,
        is_conditional: bool,
        condition_text: str,
        init_body: Optional[List[ast.stmt]] = None,
    ) -> List[ModuleAssignment]:
        """Process a single assignment statement."""
        results = []

        # Handle make_layers() calls — both tuple unpacking and direct assignment
        call_node = None
        attr_name = None

        if isinstance(stmt.value, ast.Call):
            call_name = _get_call_class_name(stmt.value)
            if call_name == "make_layers":
                call_node = stmt.value
                # Tuple unpacking: self.layers, self.start_layer, self.end_layer = make_layers(...)
                if (
                    len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Tuple)
                ):
                    for elt in stmt.targets[0].elts:
                        if (
                            isinstance(elt, ast.Attribute)
                            and isinstance(elt.value, ast.Name)
                            and elt.value.id == "self"
                        ):
                            attr_name = elt.attr
                            break
                # Direct: self.layers = make_layers(...)
                elif (
                    len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Attribute)
                    and isinstance(stmt.targets[0].value, ast.Name)
                    and stmt.targets[0].value.id == "self"
                ):
                    attr_name = stmt.targets[0].attr

        if call_node and attr_name:
            num_layers_arg = ""
            if call_node.args:
                num_layers_arg = ast.unparse(call_node.args[0])

            # Try lambda body first
            constructed_class = _extract_lambda_body_class(call_node)
            dispatch_map = None

            # Try function reference (e.g. make_layers(N, get_layer, ...))
            if not constructed_class and init_body:
                func_ref = _extract_func_ref_name(call_node)
                if func_ref:
                    local_func = _find_local_func(init_body, func_ref)
                    if local_func:
                        # Check for dispatch dict pattern
                        dispatch_map = _detect_dispatch_in_func(local_func, self.tree)
                        if dispatch_map:
                            # Use first class as representative
                            constructed_class = next(iter(dispatch_map.values()))
                        else:
                            # Direct class construction in return
                            func_classes = _extract_classes_from_func(local_func)
                            if func_classes:
                                constructed_class = func_classes[0]
                                if len(func_classes) > 1:
                                    dispatch_map = {
                                        f"variant_{i}": c
                                        for i, c in enumerate(func_classes)
                                    }

            if constructed_class:
                results.append(ModuleAssignment(
                    attr_name=attr_name,
                    class_name=constructed_class,
                    raw_args=f"make_layers({num_layers_arg}, ...)",
                    line_no=stmt.lineno,
                    is_conditional=is_conditional,
                    condition_text=condition_text,
                    dispatch_map=dispatch_map,
                ))
            return results

        # Handle nn.ModuleList([ClassName(...) for ... in range(...)]) pattern
        if isinstance(stmt.value, ast.Call):
            call_name = _get_call_class_name(stmt.value)
            if call_name == "nn.ModuleList" and stmt.value.args:
                first_arg = stmt.value.args[0]
                if isinstance(first_arg, ast.ListComp) and isinstance(first_arg.elt, ast.Call):
                    inner_class = _get_call_class_name(first_arg.elt)
                    if inner_class and _looks_like_module(inner_class, known_classes):
                        for target in stmt.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                range_arg = ""
                                if first_arg.generators:
                                    gen = first_arg.generators[0]
                                    if isinstance(gen.iter, ast.Call):
                                        range_arg = ast.unparse(gen.iter)
                                raw_args = f"nn.ModuleList([{inner_class}(...) for ... in {range_arg}])"
                                results.append(ModuleAssignment(
                                    attr_name=target.attr,
                                    class_name=inner_class,
                                    raw_args=raw_args,
                                    line_no=stmt.lineno,
                                    is_conditional=is_conditional,
                                    condition_text=condition_text,
                                ))
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
            child_node.dispatch_map = assignment.dispatch_map
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
            import re
            m = re.search(r'range\(([^)]+)\)', assignment.raw_args)
            if m:
                count_arg = m.group(1).rsplit(",", 1)[-1].strip()
                return f"x N ({count_arg})"
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
        "num_experts", "shared_expert_intermediate_size",
        "n_group", "topk_group", "topk_method",
        "first_k_dense_replace", "moe_layer_freq",
        "full_attention_interval", "layers_block_type",
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "q_lora_rank", "kv_lora_rank",
        "head_dim", "partial_rotary_factor",
        "linear_key_head_dim", "linear_value_head_dim",
        "linear_num_key_heads", "linear_num_value_heads",
        "linear_conv_kernel_dim",
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
            if child.dispatch_map and len(child.dispatch_map) > 1:
                other_classes = [c for c in child.dispatch_map.values() if c != child.class_name]
                if other_classes:
                    label += f" | {other_classes[0]}"
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
        class_trees: Optional[Dict[str, ModuleTree]] = None,
    ) -> str:
        """Format as a PyTorch-profiler-style expanded tree with instance indices.

        Expands repeated layers (make_layers) into individual instances with _0, _1, ...
        and resolves conditional branches per-layer using config metadata.
        """
        # Store pre-built class trees for dispatch resolution
        self._class_trees = class_trees or {}

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
                    if child.dispatch_map:
                        # Hybrid dispatch: resolve class per layer
                        for idx in range(num_layers):
                            resolved = self._resolve_dispatch_for_layer(
                                child, idx, config_data
                            )
                            expanded.append((resolved, idx, True))
                    else:
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

    def _resolve_dispatch_for_layer(
        self,
        child: ModuleTree,
        layer_idx: int,
        config_data: Dict[str, Any],
    ) -> ModuleTree:
        """Resolve a dispatch_map node to the correct class for a given layer index.

        Handles the Qwen3-Next hybrid pattern where full_attention_interval determines
        which layers use "attention" vs "linear_attention".
        """
        dispatch_map = child.dispatch_map
        if not dispatch_map:
            return child

        # Determine layer type key
        layer_type_key = self._get_layer_type_key(layer_idx, config_data, dispatch_map)

        resolved_class = dispatch_map.get(layer_type_key, child.class_name)

        # Create a new ModuleTree with the resolved class
        # We need to look up children from the class_map via the builder
        # For now, just swap the class_name — children will be resolved at recursion time
        resolved = ModuleTree(
            class_name=resolved_class,
            attr_name=child.attr_name,
            children=self._get_children_for_class(resolved_class, child),
            multiplicity="",
            is_conditional=False,
            condition_text="",
            line_no=child.line_no,
            dispatch_map=None,
        )
        return resolved

    def _get_children_for_class(
        self, class_name: str, original: ModuleTree
    ) -> List[ModuleTree]:
        """Get children for a resolved dispatch class.

        If the original node already has children from the hierarchy builder matching
        this class, reuse them. Otherwise return empty (leaf node in tree).
        """
        # If the hierarchy builder already resolved children for the original node's class,
        # and this is the same class, reuse them
        if original.class_name == class_name:
            return original.children
        # For dispatch-resolved nodes, we need the hierarchy builder to have built
        # child trees for all dispatch target classes. This is handled by building
        # a lookup in format_profiler_tree.
        if hasattr(self, '_class_trees') and class_name in self._class_trees:
            return self._class_trees[class_name].children
        return []

    def _get_layer_type_key(
        self, layer_idx: int, config_data: Dict[str, Any], dispatch_map: Dict[str, str]
    ) -> str:
        """Determine the dispatch key for a given layer index.

        Supports:
        - full_attention_interval: every Nth layer is "attention", rest are "linear_attention"
        - layers_block_type: explicit per-layer list from config
        """
        # Check explicit per-layer list
        block_types = config_data.get("layers_block_type")
        if isinstance(block_types, list) and layer_idx < len(block_types):
            return block_types[layer_idx]

        # Check full_attention_interval pattern (Qwen3-Next)
        interval = config_data.get("full_attention_interval")
        if isinstance(interval, int) and interval > 0:
            if (layer_idx + 1) % interval == 0:
                # This maps to "attention" key in ALL_DECODER_LAYER_TYPES
                if "attention" in dispatch_map:
                    return "attention"
                return "full_attention"
            else:
                if "linear_attention" in dispatch_map:
                    return "linear_attention"
                return "linear_attention"

        # Default: first key
        return next(iter(dispatch_map))

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


def _walk_tree(node: ModuleTree):
    """Yield all nodes in a ModuleTree."""
    yield node
    for child in node.children:
        yield from _walk_tree(child)


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

            # Pre-build class trees for all dispatch target classes
            class_trees = {}
            for node in _walk_tree(tree):
                if node.dispatch_map:
                    for cls_name in node.dispatch_map.values():
                        if cls_name not in class_trees:
                            class_trees[cls_name] = builder.build(cls_name)

            output_parts.append(
                formatter.format_profiler_tree(
                    tree, config=config, max_depth=args.depth,
                    class_trees=class_trees,
                )
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
