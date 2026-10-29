from networkx import DiGraph
from pyformlang.cfg import CFG, Terminal, Variable
from pyformlang.finite_automaton import Symbol
from scipy.sparse import csr_matrix, eye

from project.task3 import AdjacencyMatrixFA
from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    grammar = cfg_to_weak_normal_form(cfg)
    graph_fa = AdjacencyMatrixFA.from_graph(graph, start_nodes, final_nodes)
    size = graph_fa.num_states

    matrices = {
        variable: csr_matrix((size, size), dtype=bool) for variable in grammar.variables
    }
    binary_productions: list[tuple[Variable, Variable, Variable]] = []

    for production in grammar.productions:
        body = production.body
        if len(body) == 0:
            matrices[production.head] = matrices[production.head] + eye(
                size, format="csr", dtype=bool
            )
        elif len(body) == 1 and isinstance(body[0], Terminal):
            graph_matrix = graph_fa.matrices.get(Symbol(body[0].value))
            if graph_matrix is not None:
                matrices[production.head] = matrices[production.head] + graph_matrix
        elif len(body) == 2:
            binary_productions.append((production.head, body[0], body[1]))

    changed = True
    while changed:
        changed = False
        for head, left, right in binary_productions:
            product = matrices[left] @ matrices[right]
            updated = matrices[head] + product
            if updated.nnz > matrices[head].nnz:
                matrices[head] = updated
                changed = True

    result = set()
    start_matrix = matrices[grammar.start_symbol]
    rows, columns = start_matrix.nonzero()
    for source, target in zip(rows, columns, strict=True):
        if source in graph_fa.start_indices and target in graph_fa.final_indices:
            result.add(
                (
                    graph_fa.states[source].value,
                    graph_fa.states[target].value,
                )
            )
    return result
