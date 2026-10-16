from networkx import MultiDiGraph
from scipy.sparse import csr_matrix

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    query = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    if not graph_fa.start_indices or not graph_fa.final_indices:
        return set()

    sources = list(graph_fa.start_indices)
    shape = (len(sources), graph_fa.num_states)
    initial = csr_matrix(
        ([True] * len(sources), (list(range(len(sources))), sources)),
        shape=shape,
        dtype=bool,
    )

    frontier = [csr_matrix(shape, dtype=bool) for _ in query.states]
    for state in query.start_indices:
        frontier[state] = initial.copy()
    visited = [matrix.copy() for matrix in frontier]

    transitions = []
    for symbol in query.matrices.keys() & graph_fa.matrices.keys():
        rows, columns = query.matrices[symbol].nonzero()
        transitions.extend(
            (source, target, symbol)
            for source, target in zip(rows, columns, strict=True)
        )

    while any(matrix.nnz for matrix in frontier):
        reached = [csr_matrix(shape, dtype=bool) for _ in query.states]
        for source, target, symbol in transitions:
            if frontier[source].nnz:
                reached[target] = (
                    reached[target] + frontier[source] @ graph_fa.matrices[symbol]
                )
        for state in range(query.num_states):
            frontier[state] = reached[state] != reached[state].multiply(
                visited[state]
            )
            visited[state] = visited[state] + frontier[state]

    accepted = csr_matrix(shape, dtype=bool)
    for state in query.final_indices:
        accepted = accepted + visited[state]

    result = set()
    for source_row, source in enumerate(sources):
        targets = set(accepted.getrow(source_row).indices) & graph_fa.final_indices
        for target in targets:
            result.add(
                (
                    graph_fa.states[source].value,
                    graph_fa.states[target].value,
                )
            )
    return result
