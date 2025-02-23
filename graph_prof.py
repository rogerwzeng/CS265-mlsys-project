"""
Graph profiling module for PyTorch computation graphs.

This module provides profiling capabilities for FX graphs on:
- Memory usage tracking
- Execution time measurement
- Access pattern analysis
- Statistics aggregation

It works in conjunction with graph_tracer.py for graph compilation
and benchmarks.py for experiment set-up and execution.

Example Usage:
    profiler = GraphProfiler(graph_module)
    profiler.run_profiling(
        *input_args,
        warmup_steps=5,
        measure_steps=10
    )
"""

from enum import Enum
import torch
import torch.fx as fx
from typing import Dict, Any, Set

from dataclasses import dataclass
from torch.cuda import Event


class NodeType(Enum):
    """ type of the tensors in the graph """
    PARAM = 0
    ACT = 1  # activation/feature map
    GRAD = 2  # gradient
    OPT = 3  # optimizer
    OTHER = 4


@dataclass
class NodeProfile:
    """Profile data for a single graph node."""
    node_type: NodeType = NodeType.OTHER
    first_fwd_access: int = -1
    last_fwd_access: int = -1
    first_bwd_access: int = -1
    last_bwd_access: int = -1
    cuda_time: float = 0.0
    memory_consumed: int = 0
    swap_in_time: float = 0.0
    swap_out_time: float = 0.0


# ------------------------------------
# Graph Profiler performs graph execution by running the graph node by node.
#
# ------------------------------------
#
# Perform the static analysis of the graph in GraphProfiler class
# In particular, find the intermediate nodes/activations/feature_maps 
# in the graph that will be defined as
# those nodes which are not parameters (not placeholder node types) but
# are created during the forward pass and are also used in the backward
# pass for computation.
#
# The boundary between the forward pass and backward pass can be
# identified by locating the node:
#
# '%sep : [num_users=1] = # call_function[target=torch.ops.separator.sep.default]' 
#
# which will define the end of the forward pass. You will see the loss function
# after this operation and then you will encounter a node named,
#
# '%sep_backward : [num_users=1] = # call_function[target=torch.ops.separator.sep_backward.default]'
#
# This node marks the beginning of the backward pass.
#
# For intermediate nodes in the graph, you will record their last
# use in the forward pass and their first use in the backward pass.
#
# The parameters of the models are the placeholder (input) nodes of the
# graph. Note that not all the placeholder nodes of the graph are
# parameters. The optimizer's states and the input mini-batch are also
# placeholder nodes that given as inputs to the graph.
#
# The parameters and gradients of the model can be otained using the
# optimizer node's arguments. The optimizer node can be identified by the node
#
# '%_fused_adam : [num_users=3] = call_function[target=torch.ops.aten._fused_adam.default]'
#
# The argument at position 0 is the list of parameter nodes, while the
# argument at position 1 is the list of gradient nodes.
#
# Printing the input nodes, node users and node names.

class GraphProfiler(fx.Interpreter):
    """
    mu-Two Profiler: extended torch.fx Interpreter as profiler for FX computation graphs.

    Tracks memory usage, timing, and access patterns for each node
    in the computation graph during execution.
    """
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # Initialize profiling attributes
        self.node_profiles: Dict[str, NodeProfile] = {}
        self.current_step = 0
        self.is_backward = False

        # Static analysis results
        self.sep_forward_node = None
        self.sep_backward_node = None
        self.parameter_nodes: Set[fx.Node] = set()
        self.intermediate_nodes: Set[fx.Node] = set()
        self.gradient_nodes: Set[fx.Node] = set()
        self.optimizer_node = None

        # Initialize CUDA events for timing
        self.start_event = Event(enable_timing=True)
        self.end_event = Event(enable_timing=True)
        # self.swap_in_start = Event(enable_timing=True)
        # self.swap_in_end = Event(enable_timing=True)
        # self.swap_out_start = Event(enable_timing=True)
        # self.swap_out_end = Event(enable_timing=True)

        self._analyze_graph()

    def _analyze_graph(self):
        """
        Perform static analysis of the computation graph:
        - Forward/backward pass boundaries
        - Parameter nodes
        - Intermediate activations
        - Gradient nodes
        - Optimizer operations
        """
        # node buckets
        forward_nodes = []
        loss_nodes = []
        backward_nodes = []
        current_phase = 'FORWARD'

        for node in self.module.graph.nodes:
            # Initialize node profiles
            self.node_profiles[node.name] = NodeProfile()

            # Identify separator nodes and phase transitions
            if node.op == 'call_function':
                if node.target == torch.ops.separator.sep.default:
                    self.sep_forward_node = node
                    current_phase = 'LOSS'
                elif node.target == torch.ops.separator.sep_backward.default:
                    self.sep_backward_node = node
                    current_phase = 'BACKWARD'
                elif node.target == torch.ops.aten._fused_adam.default:
                    self.optimizer_node = node
                    # Extract parameter and gradient nodes from optimizer args
                    if node.args:
                        self.parameter_nodes.update(node.args[0])
                        self.gradient_nodes.update(node.args[1])

            # Collect nodes by phase
            if current_phase == 'FORWARD':
                forward_nodes.append(node)
            elif current_phase == 'LOSS':
                loss_nodes.append(node)
            else:
                backward_nodes.append(node)

            # parameter nodes (placeholders)
            if node.op == 'placeholder':
                # Parameters will be in optimizer state
                if self.optimizer_node and node in self.parameter_nodes:
                    self.node_profiles[node.name].node_type = NodeType.PARAM

            # Debug info
            print(f"\nNode Analysis: {node.name}")
            print(f"  Operation: {node.op}")
            print(f"  Target: {node.target}")
            print(f"  Inputs: {[n.name for n in node.all_input_nodes]}")
            print(f"  Users: {[n.name for n in node.users]}")

        # intermediate nodes (activations/feature maps)
        for node in forward_nodes:
            if (node not in self.parameter_nodes and
                    any(user in backward_nodes for user in node.users)):
                self.intermediate_nodes.add(node)
                self.node_profiles[node.name].node_type = NodeType.ACT

        # gradient nodes
        for node in self.gradient_nodes:
            self.node_profiles[node.name].node_type = NodeType.GRAD

        print("\nStatic Analysis Summary:")
        print(f"Forward pass nodes: {len(forward_nodes)}")
        print(f"Loss computation nodes: {len(loss_nodes)}")
        print(f"Backward pass nodes: {len(backward_nodes)}")
        print(f"Parameter nodes: {len(self.parameter_nodes)}")
        print(f"Intermediate nodes: {len(self.intermediate_nodes)}")
        print(f"Gradient nodes: {len(self.gradient_nodes)}")

    def run(self, *args, initial_env: Dict[fx.Node, Any] | None = None,
            enable_io_processing: bool = True) -> Any:
        """Execute the graph while collecting profiling data."""
        torch.cuda.synchronize()
        self.current_step += 1

        return super().run(*args, initial_env=initial_env,
             enable_io_processing=enable_io_processing)

    def run_node(self, n: fx.Node) -> Any:
        """Execute and profile a single node."""
        # During the backward pass, if a feature maps 'x' was swapped out,
        # and if node 'n' will use 'x' as one of its inputs,
        # then swap 'x' back to the GPU memory here.
        #
        # During the forward pass, if the current node 'n' is
        # the last user of a feature map 'x', then 'x' should be swapped out
        # to the CPU memory here.
        #
        # measure the run-time of a node here with CUDA Events

        profile_data = self.node_profiles[n.name]

        # Handle feature map memory swapping
        if self.is_backward:
            # Check if any input nodes are intermediate nodes that need to be swapped in
            for input_node in n.all_input_nodes:
                if input_node in self.intermediate_nodes:
                    input_tensor = self.env[input_node]  # Get tensor from environment
                    if not input_tensor.is_cuda:  # If tensor is on CPU
                        # Measure swap-in time
                        self.start_event.record()
                        self.env[input_node] = input_tensor.cuda()  # Swap to GPU
                        self.end_event.record()
                        torch.cuda.synchronize()
                        profile_data.swap_in_time += self.start_event.elapsed_time(self.end_event)

        # static access pattern
        if not self.is_backward:
            if profile_data.first_fwd_access == -1:
                profile_data.first_fwd_access = self.current_step
            profile_data.last_fwd_access = self.current_step
        else:
            if profile_data.first_bwd_access == -1:
                profile_data.first_bwd_access = self.current_step
            profile_data.last_bwd_access = self.current_step

        # Measure execution time
        self.start_event.record()
        result = super().run_node(n)
        self.end_event.record()

        torch.cuda.synchronize()
        profile_data.cuda_time += self.start_event.elapsed_time(self.end_event)

        # memory consumption
        if isinstance(result, torch.Tensor) and result.is_cuda:
            profile_data.memory_consumed = result.element_size() * result.nelement()

        # Handle swapping out feature maps in forward pass
        if not self.is_backward and n in self.intermediate_nodes:
            # Check if this is the last use of this node in forward pass
            is_last_forward_use = all(
                user in self.gradient_nodes or  # Used only in backward pass
                user == self.sep_forward_node   # Or used by separator
                for user in n.users
            )

            if is_last_forward_use:
                # Measure swap-out time
                self.start_event.record()
                self.env[n] = result.cpu()  # Swap to CPU
                self.end_event.record()
                torch.cuda.synchronize()
                profile_data.swap_out_time += self.start_event.elapsed_time(self.end_event)

        """
        # Log profiling data to WandB
        wandb.log({
            f"{n.name}/first_fwd_access": profile_data.first_fwd_access,
            f"{n.name}/last_fwd_access": profile_data.last_fwd_access,
            f"{n.name}/first_bwd_access": profile_data.first_bwd_access,
            f"{n.name}/last_bwd_access": profile_data.last_bwd_access,
            f"{n.name}/cuda_time": profile_data.cuda_time,
            f"{n.name}/memory_consumed": profile_data.memory_consumed,
        }) """

        return result

    def aggregate_stats(self, warmup_steps: int = 5, measure_steps: int = 10) -> None:
        """
        Aggregate profiling statistics after warm-up iterations.
        Run x warm-up and y actual steps, with run-time measurements averaged over y runs.
        
        Args:
            warmup_steps: Number of warm-up iterations to discard
            measure_steps: Number of measurement iterations to average over
        """
        total_memory = sum(p.memory_consumed for p in self.node_profiles.values())
        total_time = sum(p.cuda_time for p in self.node_profiles.values())

        # Average measurements over # of measurement steps
        if measure_steps > 0:
            total_time /= measure_steps

        print(f"\nAggregate Statistics (after {warmup_steps} warm-up steps, averaged over {measure_steps} steps):")
        print(f"Total memory: {total_memory / 1e6:.2f} MB")
        print(f"Total CUDA time: {total_time:.2f} ms")

        """
        wandb.log({
            "total_memory": total_memory,
            "total_time": total_time
        })"""

    def print_stats(self) -> None:
        """Print detailed statistics for each node."""
        print("\nPer-Node Statistics:")
        for node_name, profile_data in self.node_profiles.items():
            print(f"\nNode: {node_name}")
            print(f"First forward access: {profile_data.first_fwd_access}")
            print(f"Last forward access: {profile_data.last_fwd_access}")
            print(f"First backward access: {profile_data.first_bwd_access}")
            print(f"Last backward access: {profile_data.last_bwd_access}")
            print(f"CUDA time: {profile_data.cuda_time:.2f} ms")
            print(f"Memory used: {profile_data.memory_consumed / 1e6:.2f} MB")
            if profile_data.swap_in_time > 0 or profile_data.swap_out_time > 0:
                print(f"Swap in time: {profile_data.swap_in_time:.2f} ms")
                print(f"Swap out time: {profile_data.swap_out_time:.2f} ms")

    def reset_stats(self) -> None:
        """
        Reset all profiling statistics.
        Called after warm-up iterations and before measurement iterations.
        """
        for profile in self.node_profiles.values():
            profile.__init__()   # Reset values
            """
            profile.first_fwd_access = -1
            profile.last_fwd_access = -1
            profile.first_bwd_access = -1
            profile.last_bwd_access = -1
            profile.cuda_time = 0.0
            profile.memory_consumed = 0
            # profile.swap_in_time = 0.0
            # profile.swap_out_time = 0.0
            """
        self.current_step = 0
        self.is_backward = False

    def run_profiling(self, *args, warmup_steps: int = 5, measure_steps: int = 10) -> None:
        """
        Run complete profiling sequence with warm-up and measurement phases.
        
        Args:
            *args: Arguments to pass to the graph module
            warmup_steps: Number of warm-up iterations
            measure_steps: Number of measurement iterations
        """
        # Warm-up phase
        print(f"\nStarting {warmup_steps} warm-up iterations...")
        for i in range(warmup_steps):
            self.run(*args)

        # Reset statistics after warm-up
        self.reset_stats()

        # Measurement phase
        print(f"\nStarting {measure_steps} measurement iterations...")
        for i in range(measure_steps):
            self.run(*args)

        # Aggregate and print results
        self.aggregate_stats(warmup_steps, measure_steps)
