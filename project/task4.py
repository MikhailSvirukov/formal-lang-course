from networkx import MultiDiGraph
from scipy.sparse import csr_matrix

from project.task3 import AdjacencyMatrixFA, intersect_automata


def _union_transition_matrices(automaton: AdjacencyMatrixFA) -> csr_matrix:
    adjacency = csr_matrix((automaton.num_states, automaton.num_states), dtype=bool)
    for matrix in automaton.matrices.values():
        adjacency = adjacency.maximum(matrix)
    return adjacency


def _new_frontier(candidates: csr_matrix, visited: csr_matrix) -> csr_matrix:
    already_visited = candidates.multiply(visited)
    result = candidates.astype("int8") - already_visited.astype("int8")
    result.eliminate_zeros()
    return result.astype(bool)


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    graph_automaton = AdjacencyMatrixFA.from_graph(graph, start_nodes, final_nodes)
    regex_automaton = AdjacencyMatrixFA.from_regex(regex)
    product = intersect_automata(graph_automaton, regex_automaton)

    sources = [state.value for state in graph_automaton.start_states]
    source_to_row = {source: row for row, source in enumerate(sources)}

    rows = []
    columns = []
    for product_index in product.start_indices:
        graph_state, _ = product.states[product_index]
        rows.append(source_to_row[graph_state.value])
        columns.append(product_index)

    frontier = csr_matrix(
        ([True] * len(rows), (rows, columns)),
        shape=(len(sources), product.num_states),
        dtype=bool,
    )
    visited = frontier.copy()
    adjacency = _union_transition_matrices(product)

    while frontier.nnz:
        candidates = (frontier @ adjacency).astype(bool)
        frontier = _new_frontier(candidates, visited)
        visited = visited.maximum(frontier)

    result = set()
    for final_index in product.final_indices:
        target_state, _ = product.states[final_index]
        reached_by, _ = visited[:, final_index].nonzero()
        for source_row in reached_by:
            result.add((sources[source_row], target_state.value))
    return result
