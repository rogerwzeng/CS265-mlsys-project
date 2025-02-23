"""
Graph tracing and compilation module for PyTorch models.

This module provides functionality for tracing, compiling, and optimizing PyTorch
computation graphs. It includes utilities for:
- Separating forward and backward passes
- Graph compilation and transformation
- Gradient tracking and optimization
- State management for compiled functions

The module is designed to work with PyTorch's FX graph system and supports
both training and optimization operations.

Key components:
- SEPFunction: Custom autograd function for pass separation
- compile: Main function for compiling and optimizing computation graphs
- _compile: Internal compilation implementation
"""

from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Union
from utils import SPMD_DECOMP_TABLE

import torch

# We need to import _functional_collectives to trigger op registration
#import torch.distributed._functional_collectives
import torch.nn as nn
import torch.optim as optim
import torch.utils._pytree as pytree
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
'''
from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.placement_types import DTensorSpec
'''
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph import CodeGen, _PyTreeCodeGen, _PyTreeInfo
from torch.nn.utils import stateless
from torch.utils.hooks import RemovableHandle


DEBUG_MODE = False  # for __compile()


def sep(x: torch.Tensor) -> torch.Tensor:
    """
    Identity function used as a separator marker in computation graphs.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Unmodified input tensor
    """
    return x


def sep_backward(grad: torch.Tensor) -> torch.Tensor:
    """
    Identity function for backward pass gradient propagation.
    
    Args:
        grad (torch.Tensor): Input gradient tensor
        
    Returns:
        torch.Tensor: Unmodified gradient tensor
    """
    return grad


separator_lib = torch.library.Library("separator", "DEF")
separator_lib.define("sep(Tensor x) -> Tensor")
separator_lib.impl("sep", sep, "CompositeExplicitAutograd")
separator_lib.define("sep_backward(Tensor x) -> Tensor")
separator_lib.impl("sep_backward", sep_backward, "CompositeExplicitAutograd")

'''
def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
    (x,) = op_schema.args_schema
    assert isinstance(x, DTensorSpec), f"expecting DTensorSpec but got {x}"

    return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))

def _prop_sepm(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)

def _prop_sepm_backward(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)

DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(torch.ops.separator.sep.default, _prop_sepm)
DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(torch.ops.separator.sep_backward.default, _prop_sepm_backward)
'''


class SEPFunction(torch.autograd.Function):
    """
    Custom autograd function that acts as a separator between forward and backward passes.
    
    Used to mark boundaries in the computation graph for analysis and optimization.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass identity operation.
        
        Args:
            ctx: Autograd context
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Unmodified input tensor
        """
        return torch.ops.separator.sep(x)


    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:
        """
        Backward pass identity operation.
        
        Args:
            ctx: Autograd context
            grad_x (torch.Tensor): Input gradient tensor
            
        Returns:
            torch.Tensor: Unmodified gradient tensor
        """
        return torch.ops.separator.sep_backward(grad_x)


# Dummy op used by data parallel to tag gradients.
_spmd_lib_def = torch.library.Library("dummy", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("dummy", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")


class _PyTreeCodeGenOutputsOnly(_PyTreeCodeGen):
    # pyre-ignore[3]
    def process_inputs(self, *args: Any) -> Any:
        return args

    # pyre-ignore[2, 3]
    def gen_fn_def(self, free_vars, maybe_return_annotation):
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: fx.GraphModule) -> fx.GraphModule:
    """Move the responsibility of flattening the input arguments from the
    graph module to the caller.

    Example:

        output = gm(my_struct)

        gm = gm(to_caller_flattened_graph_module)

        output = gm(*pytree.flatten(my_struct)[0])
    """
    # pyre-ignore[16]
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            # pyre-ignore[6]
            orig_args=None,  # type: ignore[arg-type]
            # pyre-ignore[6]
            in_spec=None,  # type: ignore[arg-type]
            # pyre-ignore[16]
            out_spec=gm._graph._codegen.pytree_info.out_spec,
        )
    )
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

@contextmanager
def gradients_tagging(params: Dict[str, nn.Parameter]):
    """
    This is a helper function that tags the gradient of the parameters
    with a special tag, so that we can identify them during SPMD expansion.

    Installs temporary hooks on parameters to mark their gradients during
    backward pass computation.  It's safe to trace those hooks and we would 
    remove those nodes later.
    
    Args:
        params (Dict[str, nn.Parameter]): Dictionary of named parameters
        
    Yields:
        None: Executes the wrapped code block with gradient tagging active
    """

    tagging_hooks: List[RemovableHandle] = []
    try:
        for p in params.values():
            h = p.register_hook(lambda grad: torch.ops.dummy.tag_grad(grad))
            tagging_hooks.append(h)
        yield
    finally:
        # remove those hooks after tracing
        for h in tagging_hooks:
            h.remove()


@contextmanager
def _rematerialize_optimizer(
    opt: optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, nn.Parameter],
):
    """
    Context manager for optimizer state rematerialization.
    Temporarily updates optimizer state with proxy tensors for tracing.
    Args:
        opt (optim.Optimizer): The optimizer instance
        named_states (Dict[str, Any]): Named optimizer states
        params (Dict[str, nn.Parameter]): Model parameters
    """
    assert opt is not None

    # update opt.state with proxy tensors
    orig_states = copy(opt.state)
    for n in named_states:
        # opt.state's key type is string, but optimizer uses Parameter as keys
        opt.state[params[n]] = named_states[n]  # type: ignore[index]

    # FIXME: support multiple parameter groups
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()

    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state = orig_states


@contextmanager
def _enable_compile():
    # The return value of torch._utils.is_compiling changes optimizer behavior.
    # We need that function to return True to include optimizer in the graph.
    # See: https://github.com/pytorch/pytorch/blob/a524123c91ab399c9dd6882c1189596dd77e7734/torch/optim/optimizer.py#L41
    def f_true():
        return True

    orig_is_compiling_code = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code


@dataclass
class _CompiledResult:
    """
    Container for compiled graph module and associated state.
    
    Attributes:
        gm (fx.GraphModule): Compiled graph module
        mod (nn.Module): Original module
        opt (Optional[torch.optim.Optimizer]): Associated optimizer
        flat_state (List[torch.Tensor]): Flattened state tensors
    """
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]


def _compile(func: Callable, *args: Any, **kwargs: Any) -> _CompiledResult:
    """
    Internal compilation function that traces and processes a callable.
    
    Creates a graph module from the function and processes it for optimization.
    
    Args:
        func (Callable): Function to compile
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        _CompiledResult: Compiled graph module and associated state
        
    Raises:
        AssertionError: If no nn.Module instance is found in arguments
    """
    # 1. Extract nn.Module and Optimizer from args and kwargs
    mod, opt = None, None
    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, nn.Module):
            assert mod is None, "Only support single nn.Module for now"
            mod = arg
        if isinstance(arg, optim.Optimizer):
            assert opt is None, "Only support single Optimizer for now"
            opt = arg
    assert mod is not None, "Couldn't find nn.Module instances from the arguments."

    # 2. Trace the stateless version of the train_step
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(mod.named_buffers(remove_duplicate=False))

    named_states: Dict[str, nn.Parameter] = {}
    # Pass named_states instead of opt.state to stateless_func, because
    # the later uses nn.Parameter as key. During tracing, we need to
    # make sure optimizers can find the states using proxy tensors.
    for n, p in params.items():
        if p in opt.state:
            # opt.state's key type is string, but optimizer uses
            # Parameter as keys
            named_states[n] = opt.state[p]

    # Lift states and parameters as function arguments so that make_fx
    # can trace operations applied to them

    def stateless_func(
        func: Callable,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, nn.Parameter],
        args: Any,
        kwargs: Any,
    ) -> tuple[Any, List[nn.Parameter], List[Any]]:
        """
        Creates a stateless version of a function for tracing purposes.
        
        This function temporarily replaces module parameters and optimizer states
        with proxy tensors to enable tracing of the computation graph. It manages
        both model state (parameters and buffers) and optimizer state during the
        tracing process.
        
        Args:
            func (Callable): The original function to make stateless
            params (Dict[str, nn.Parameter]): Named model parameters
            buffers (Dict[str, torch.Tensor]): Named model buffers
            named_states (Dict[str, nn.Parameter]): Named optimizer states
            args (Any): Original function's positional arguments
            kwargs (Any): Original function's keyword arguments
            
        Returns:
            tuple:
                - Any: Original function's return value
                - List[nn.Parameter]: Updated model parameters
                - List[Any]: Updated optimizer states
                
        Example:
            >>> def train_step(model, optimizer, batch):
            ...     loss = model(batch).sum()
            ...     loss.backward()
            ...     optimizer.step()
            >>> 
            >>> params = dict(model.named_parameters())
            >>> buffers = dict(model.named_buffers())
            >>> named_states = {n: optimizer.state[p] for n, p in params.items()}
            >>> ret, updated_params, updated_states = stateless_func(
            ...     train_step, params, buffers, named_states, 
            ...     (model, optimizer, batch), {}
            ... )
        
        Notes:
            - Uses context managers to temporarily modify module and optimizer state
            - Installs gradient tagging hooks for SPMD expansion
            - Preserves original state after execution
            - Enables tracing of both forward and backward passes
            - Handles optimizer state rematerialization
        """

        with stateless._reparametrize_module(
            mod, {**params, **buffers}
        ), _rematerialize_optimizer(
            opt, named_states, params
        ) if opt else nullcontext():
            # Installing hooks onto gradients to identify the gradients.
            with gradients_tagging(params):
                ret = func(*args, **kwargs)

            # the return value of the function must be the original return value
            # updated paramaters and updated optimizer states
            return ret, list(mod.parameters()), list(named_states.values())

    tracing_mode = "fake"
    fake_mode = FakeTensorMode()

    def _get_fake_args(arg: torch.Tensor) -> torch.Tensor:
        fake_arg = fake_mode.from_tensor(arg)
        return fake_arg

    args = pytree.tree_map_only(torch.Tensor, _get_fake_args, args)
    kwargs = pytree.tree_map_only(torch.Tensor, _get_fake_args, kwargs)


    # turn on DEBUG_MODE for debug, turn off for production
    compile_context = (
        torch.autograd.detect_anomaly(check_nan=False)
        if DEBUG_MODE
        else nullcontext()
    )

    # Actual compile starts
    with _enable_compile(), compile_context:
        gm = make_fx(
            partial(stateless_func, func),
            tracing_mode=tracing_mode,
            decomposition_table=SPMD_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(params, buffers, named_states, args, kwargs)
        """
        https://pytorch.org/docs/stable/generated/torch.fx.experimental.proxy_tensor.make_fx.html

        Creates an FX graph by tracing a function using proxy tensors.

        This function enables advanced graph capture by using proxy tensors during tracing,
        which can track operations more comprehensively than traditional FX tracing.

        Args:
            fn (Callable): Function to be traced. Can be any Python callable that takes
                tensor inputs and performs PyTorch operations.

            tracing_mode (str, optional): Specifies the tracing mode. Options:
                - "real": Uses real tensors during tracing
                - "fake": Uses fake tensors (recommended for memory efficiency)
                - "symbolic": Uses symbolic shapes for dynamic tracing
                Default: "real"

            decomposition_table (Dict[Callable, Callable], optional): Maps operations
                to their decomposed implementations. Used to break down complex ops
                into simpler ones during tracing.
                Default: None

            _allow_non_fake_inputs (bool, optional): Internal flag to allow mixing
                of fake and real tensors during tracing. Generally should be False
                unless you know what you're doing.
                Default: False

        Returns:
            Callable: A function that when called with the same argument types as `fn`,
            returns an FX GraphModule representing the traced operations.

        Example:
            >>> def my_fun(x, y):
            ...     return torch.mm(x, y)
            >>>
            >>> traced_fn = make_fx(my_fun)
            >>> x = torch.randn(2, 3)
            >>> y = torch.randn(3, 4)
            >>> graph_module = traced_fn(x, y)
            >>> print(graph_module.graph)

        Notes:
            - More powerful than traditional FX tracing
            - Can capture dynamic control flow
            - Handles autograd operations
            - Supports complex tensor operations
            - Can trace through nn.Module methods

        Raises:
            RuntimeError: If tracing fails or encounters unsupported operations
            ValueError: If invalid tracing_mode is specified

        See Also:
            - torch.fx.symbolic_trace: Traditional FX tracing
            - torch.fx.GraphModule: The output graph module type
        """

    params_and_buffers: Dict[str, Union[torch.Tensor, nn.Parameter]] = {
        **params,
        **buffers,
    }

    flat_state, _ = pytree.tree_flatten([params_and_buffers, named_states])

    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
        if node.target == torch.ops.dummy.tag_grad.default:
            grad_node = node.all_input_nodes[0]
            node.replace_all_uses_with(grad_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)

    gm = _to_caller_flattened_graph_module(gm)

    return _CompiledResult(gm, mod, opt, flat_state)


# Note that the Python convention of __dict__ requires the key to be str.
# TODO: ensure the key is unique.
COMPILED_OBJECT_KEY = "_compiled_obj"


def compile(func: Callable, gm_transformation: Callable) -> Callable:
    """
    Main compilation function that wraps a function for optimized execution.
    
    Traces the function, applies transformations, and caches the result for
    repeated execution.
    
    Args:
        func (Callable): Function to compile
        gm_transformation (Callable): Transformation to apply to the graph module
        
    Returns:
        Callable: Wrapped function that executes the compiled graph
    """
    @wraps(func)  # Preserves the metadata of the original function
    def wrapper(*args, **kwargs):
        """
        Wrapper function that handles compilation and execution.
        
        On first call:
        - Compiles the function
        - Applies transformations
        - Caches the result
        
        On subsequent calls:
        - Uses cached compiled version
        - Processes inputs
        - Executes optimized graph
        
        Args:
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Any: Output from the compiled graph execution
        """
        # Check if we already have a compiled version stored in wrapper's dictionary
        first_iter = False
        compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
        if compiled_obj is None:
            first_iter = True
            compiled_obj = _compile(func, *args, **kwargs)
            wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
        
        # Combine two things:
        # 1. compiled_obj.flat_state: Contains model parameters, buffers, optimizer states
        # 2. Flattened version of current function arguments
        # Example:
        # If args = (model, optimizer, batch)
        # and kwargs = {'learning_rate': 0.01}
        # pytree.tree_flatten([args, kwargs])[0] would create a flat list:
        # [model, optimizer, batch, 0.01]
        flat_inps = compiled_obj.flat_state + pytree.tree_flatten([args, kwargs])[0]

        # Only apply transformation on first iteration
        if first_iter and gm_transformation:
            compiled_obj.gm = gm_transformation(compiled_obj.gm, flat_inps)

        # no storing gradients
        with torch.no_grad():
            # Execute the compiled graph, [0] because the graph might return multiple values
            output = compiled_obj.gm(*flat_inps)[0] 

        return output

    return wrapper
