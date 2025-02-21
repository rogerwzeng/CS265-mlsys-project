from enum import Enum
from typing import Dict
import torch
import torch.fx as fx
from typing import Dict, Any


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # You should perform the static analysis of the graph here. In
        # particular you might want to find the intermediate
        # nodes/activations/feature_maps in the graph that will be defined as
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
        # after thsi operation and then you will encounter a node named,
        #
        # '%sep_backward : [num_users=1] = # call_function[target=torch.ops.separator.sep_backward.default]'
        #
        # This node marks the beginning of the backward pass.
        # For intermediate nodes in the graph, you will record their last
        # use in the forward pass and their first use in the backward pass.     
        #
        # The parameters of the models are the placeholder (input) nodes of the
        # graph. Note that not all the placeholder nodes of the graph are
        # parameters. The optimizer's states and the input mini-batch are also
        # placeholder nodes that given as inputs to the graph.
        #
        # The parameters and gradients of the model can be otained using the
        # optimizer node's arguments. The optimizer node can be identified by
        # the node '%_fused_adam : [num_users=3] =
        # call_function[target=torch.ops.aten._fused_adam.default]'.
        # The argument at position 0 is the list of parameter nodes, while the
        # argument at position 1 is the list of gradient nodes.

        # Printing the input nodes, node users and node names.

        for node in self.module.graph.nodes:
            print("Node name: ", node.name)
            print("Node type: ", node.op)
            print("Node target: ", node.target)
            print("Input to this node", node.all_input_nodes)
            print("Users of this node: ", node.users)

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:
        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.

        # you can start measuring the run-time of a node here
        result = super().run_node(n)
        # you can end measuring the run-time of a node here HINT: Use
        # torch.cuda.Events for doing time measurements of operations.

        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.

        return result

    def aggregate_stats(self) -> None:
        # You are expected run the profiler for x warm-up iterations and y
        # actual measurement iterations. The run-time measurement then needs to
        # be averaged over the y runs.
        pass

    def print_stats(self) -> None:
        pass

    def reset_stats(self) -> None:
        # The statistics must be cleared out after x warm-up iterations and
        # reset before the actual measurement begins.
        pass
