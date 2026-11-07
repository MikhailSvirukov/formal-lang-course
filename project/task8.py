from networkx import DiGraph
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import csr_matrix

from project.task3 import AdjacencyMatrixFA, intersect_automata
from project.task6 import cfg_to_weak_normal_form


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    grammar = cfg_to_weak_normal_form(cfg)
    terminal_values = {terminal.value for terminal in grammar.terminals}
    used_labels = terminal_values | {variable.value for variable in grammar.variables}
    call_labels = {}
    counter = 0
    for variable in grammar.variables:
        label = variable.value
        if label in terminal_values:
            while f"__RSM_CALL_{counter}" in used_labels:
                counter += 1
            label = f"__RSM_CALL_{counter}"
            used_labels.add(label)
        call_labels[variable] = Symbol(label)

    automata = {
        variable: NondeterministicFiniteAutomaton() for variable in grammar.variables
    }
    for automaton in automata.values():
        automaton.add_start_state(0)

    for production in grammar.productions:
        automaton = automata[production.head]
        current = 0
        for symbol in production.body:
            target = len(automaton.states)
            transition = (
                call_labels[symbol]
                if isinstance(symbol, Variable)
                else Symbol(symbol.value)
            )
            automaton.add_transition(current, transition, target)
            current = target
        automaton.add_final_state(current)

    boxes = {
        Box(automaton, call_labels[variable])
        for variable, automaton in automata.items()
    }
    return RecursiveAutomaton(
        initial_label=call_labels[grammar.start_symbol], boxes=boxes
    )


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    heads = [
        line.split("->", 1)[0].strip() for line in ebnf.splitlines() if "->" in line
    ]
    if not heads:
        return RecursiveAutomaton()
    start = "S" if "S" in heads else heads[0]
    return RecursiveAutomaton.from_text(ebnf, start_symbol=Symbol(start))


def _rsm_to_matrix_automaton(
    rsm: RecursiveAutomaton,
) -> tuple[AdjacencyMatrixFA, dict[Symbol, tuple[set[int], set[int]]]]:
    states = []
    coordinates: dict[Symbol, tuple[list[int], list[int]]] = {}
    box_boundaries = {}

    for label, box in rsm.boxes.items():
        local_states = list(box.dfa.states)
        local_to_global = {
            state: len(states) + index for index, state in enumerate(local_states)
        }
        states.extend((label, state) for state in local_states)

        starts = {local_to_global[state] for state in box.dfa.start_states}
        finals = {local_to_global[state] for state in box.dfa.final_states}
        box_boundaries[label] = (starts, finals)

        for source, symbol, target in box.dfa._transition_function:
            rows, columns = coordinates.setdefault(symbol, ([], []))
            rows.append(local_to_global[source])
            columns.append(local_to_global[target])

    size = len(states)
    matrices = {
        symbol: csr_matrix(
            ([True] * len(rows), (rows, columns)),
            shape=(size, size),
            dtype=bool,
        )
        for symbol, (rows, columns) in coordinates.items()
    }
    starts = set().union(*(bounds[0] for bounds in box_boundaries.values()))
    finals = set().union(*(bounds[1] for bounds in box_boundaries.values()))

    return (
        AdjacencyMatrixFA.from_components(states, matrices, starts, finals),
        box_boundaries,
    )


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    rsm_automaton, box_boundaries = _rsm_to_matrix_automaton(rsm)
    graph_automaton = AdjacencyMatrixFA.from_graph(graph, start_nodes, final_nodes)
    graph_size = graph_automaton.num_states
    if graph_size == 0 or rsm.initial_label not in rsm.boxes:
        return set()

    for label in rsm.boxes:
        graph_automaton.matrices[label] = csr_matrix(
            (graph_size, graph_size), dtype=bool
        )

    changed = True
    while changed:
        changed = False
        product = intersect_automata(rsm_automaton, graph_automaton)
        closure = product.transitive_closure()

        for label, (box_starts, box_finals) in box_boundaries.items():
            inferred = csr_matrix((graph_size, graph_size), dtype=bool)
            for box_start in box_starts:
                row_start = box_start * graph_size
                for box_final in box_finals:
                    column_start = box_final * graph_size
                    inferred += closure[
                        row_start : row_start + graph_size,
                        column_start : column_start + graph_size,
                    ]

            known = graph_automaton.matrices.get(
                label, csr_matrix((graph_size, graph_size), dtype=bool)
            )
            updated = known + inferred
            if updated.nnz > known.nnz:
                graph_automaton.matrices[label] = updated
                changed = True

    start_symbol_matrix = graph_automaton.matrices.get(rsm.initial_label)
    if start_symbol_matrix is None:
        return set()

    result = set()
    rows, columns = start_symbol_matrix.nonzero()
    for source, target in zip(rows, columns, strict=True):
        if (
            source in graph_automaton.start_indices
            and target in graph_automaton.final_indices
        ):
            result.add(
                (
                    graph_automaton.states[source].value,
                    graph_automaton.states[target].value,
                )
            )
    return result
