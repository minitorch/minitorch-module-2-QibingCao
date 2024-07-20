from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Dict
from collections import deque

from typing_extensions import Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_list = list(vals)  # 将元组转换为列表
    vals_list[arg] -= epsilon
    f_minus = f(*vals_list)

    vals_list[arg] += 2 * epsilon
    f_plus = f(*vals_list)

    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    sorted_variables = []
    visited = set()

    def bfs(start_variable: Variable) -> None:
        dq = deque([start_variable])

        while dq:
            cur = dq.popleft()

            if cur.unique_id in visited:
                continue

            visited.add(cur.unique_id)
            sorted_variables.append(cur)

            for parent in cur.parents:
                if (
                    not parent.is_leaf()
                    and not parent.is_constant()
                    and parent.unique_id not in visited
                ):
                    dq.append(parent)

    bfs(variable)

    # for item in sorted_variables:
    #     print("\n")
    #     print(item)
    #     print(item.parents)
    #     print("\n")
    return sorted_variables


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_variables = topological_sort(variable)
    derive_table: Dict[int, float] = {variable.unique_id: deriv}

    def back_helper(vari: Variable) -> None:
        d_output = derive_table[vari.unique_id]
        parents_derive = cur_variable.chain_rule(d_output)
        for parent, derivative in parents_derive:
            if parent.is_leaf():
                parent.accumulate_derivative(derivative)
            else:
                if parent.unique_id not in derive_table:
                    derive_table[parent.unique_id] = 0.0
                derive_table[parent.unique_id] += derivative

    for cur_variable in sorted_variables:
        back_helper(cur_variable)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
