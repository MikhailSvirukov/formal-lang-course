from collections.abc import Iterable
from typing import Hashable

from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import csr_matrix, eye, kron

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton):
        states = list(automaton.states)
        state_to_index = {state: index for index, state in enumerate(states)}
        size = len(states)

        coordinates: dict[Symbol, tuple[list[int], list[int]]] = {}
        for source, symbol, target in automaton._transition_function:
            rows, columns = coordinates.setdefault(symbol, ([], []))
            rows.append(state_to_index[source])
            columns.append(state_to_index[target])

        matrices = {
            symbol: csr_matrix(
                ([True] * len(rows), (rows, columns)),
                shape=(size, size),
                dtype=bool,
            )
            for symbol, (rows, columns) in coordinates.items()
        }

        self._set_components(
            states=states,
            matrices=matrices,
            start_indices={state_to_index[state] for state in automaton.start_states},
            final_indices={state_to_index[state] for state in automaton.final_states},
        )

    @classmethod
    def from_regex(cls, regex: str) -> "AdjacencyMatrixFA":
        return cls(regex_to_dfa(regex))

    @classmethod
    def from_graph(
        cls,
        graph: MultiDiGraph,
        start_states: set[Hashable],
        final_states: set[Hashable],
    ) -> "AdjacencyMatrixFA":
        return cls(graph_to_nfa(graph, start_states, final_states))

    @classmethod
    def from_components(
        cls,
        states: list[Hashable],
        matrices: dict[Symbol, csr_matrix],
        start_indices: set[int],
        final_indices: set[int],
    ) -> "AdjacencyMatrixFA":
        result = cls.__new__(cls)
        result._set_components(states, matrices, start_indices, final_indices)
        return result

    def _set_components(
        self,
        states: list[Hashable],
        matrices: dict[Symbol, csr_matrix],
        start_indices: set[int],
        final_indices: set[int],
    ) -> None:
        self.states = states
        self.matrices = matrices
        self.state_to_index = {state: index for index, state in enumerate(states)}
        self.index_to_state = dict(enumerate(states))
        self.start_indices = start_indices
        self.final_indices = final_indices
        self.start_states = {states[index] for index in start_indices}
        self.final_states = {states[index] for index in final_indices}

    @property
    def num_states(self) -> int:
        return len(self.states)

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current = csr_matrix(
            (
                [True] * len(self.start_indices),
                ([0] * len(self.start_indices), list(self.start_indices)),
            ),
            shape=(1, self.num_states),
            dtype=bool,
        )

        for value in word:
            symbol = value if isinstance(value, Symbol) else Symbol(value)
            matrix = self.matrices.get(symbol)
            if matrix is None:
                return False
            current = current @ matrix
            if current.nnz == 0:
                return False

        return any(current[0, index] for index in self.final_indices)

    def transitive_closure(self) -> csr_matrix:
        closure = eye(self.num_states, format="csr", dtype=bool)
        for matrix in self.matrices.values():
            closure = closure.maximum(matrix)

        while True:
            expanded = closure.maximum(closure @ closure)
            expanded.eliminate_zeros()
            if expanded.nnz == closure.nnz:
                return expanded
            closure = expanded

    def is_empty(self) -> bool:
        if not self.start_indices or not self.final_indices:
            return True

        closure = self.transitive_closure()
        return not any(
            closure[start, final]
            for start in self.start_indices
            for final in self.final_indices
        )


def intersect_automata(
    automaton1: AdjacencyMatrixFA,
    automaton2: AdjacencyMatrixFA,
) -> AdjacencyMatrixFA:
    states = [
        (state1, state2) for state1 in automaton1.states for state2 in automaton2.states
    ]
    second_size = automaton2.num_states

    start_indices = {
        first * second_size + second
        for first in automaton1.start_indices
        for second in automaton2.start_indices
    }
    final_indices = {
        first * second_size + second
        for first in automaton1.final_indices
        for second in automaton2.final_indices
    }

    common_symbols = automaton1.matrices.keys() & automaton2.matrices.keys()
    matrices = {
        symbol: kron(
            automaton1.matrices[symbol],
            automaton2.matrices[symbol],
            format="csr",
        ).astype(bool)
        for symbol in common_symbols
    }

    return AdjacencyMatrixFA.from_components(
        states, matrices, start_indices, final_indices
    )


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    graph_automaton = AdjacencyMatrixFA.from_graph(graph, start_nodes, final_nodes)
    regex_automaton = AdjacencyMatrixFA.from_regex(regex)
    product = intersect_automata(graph_automaton, regex_automaton)
    reachable = product.transitive_closure()

    result = set()
    for start_index in product.start_indices:
        source, _ = product.states[start_index]
        for final_index in product.final_indices:
            if reachable[start_index, final_index]:
                target, _ = product.states[final_index]
                result.add((source.value, target.value))
    return result
