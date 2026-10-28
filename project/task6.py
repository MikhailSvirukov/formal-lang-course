from collections import defaultdict, deque

from networkx import DiGraph
from pyformlang.cfg import CFG, Production, Terminal, Variable

from project.task2 import graph_to_nfa


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    normal_form = cfg.to_normal_form()
    productions = set(normal_form.productions)

    if cfg.generate_epsilon():
        productions.add(Production(normal_form.start_symbol, []))

    return CFG(
        start_symbol=normal_form.start_symbol,
        productions=productions,
    )


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    grammar = cfg_to_weak_normal_form(cfg)
    graph_automaton = graph_to_nfa(graph, start_nodes, final_nodes)

    epsilon_heads: set[Variable] = set()
    terminal_heads: dict[object, set[Variable]] = defaultdict(set)
    binary_heads: dict[tuple[Variable, Variable], set[Variable]] = defaultdict(set)

    for production in grammar.productions:
        body = production.body
        if len(body) == 0:
            epsilon_heads.add(production.head)
        elif len(body) == 1 and isinstance(body[0], Terminal):
            terminal_heads[body[0].value].add(production.head)
        elif len(body) == 2:
            binary_heads[(body[0], body[1])].add(production.head)

    triples: set[tuple[Variable, object, object]] = set()
    starting_at: dict[object, set[tuple[Variable, object, object]]] = defaultdict(set)
    ending_at: dict[object, set[tuple[Variable, object, object]]] = defaultdict(set)
    queue: deque[tuple[Variable, object, object]] = deque()

    def add(variable: Variable, source: object, target: object) -> None:
        triple = (variable, source, target)
        if triple in triples:
            return
        triples.add(triple)
        starting_at[source].add(triple)
        ending_at[target].add(triple)
        queue.append(triple)

    for state in graph_automaton.states:
        for variable in epsilon_heads:
            add(variable, state.value, state.value)

    for source, transitions in graph_automaton.to_dict().items():
        for symbol, targets in transitions.items():
            for target in targets:
                for variable in terminal_heads[symbol.value]:
                    add(variable, source.value, target.value)

    while queue:
        variable, source, target = queue.popleft()

        for left_variable, left_source, _ in tuple(ending_at[source]):
            for head in binary_heads[(left_variable, variable)]:
                add(head, left_source, target)

        for right_variable, _, right_target in tuple(starting_at[target]):
            for head in binary_heads[(variable, right_variable)]:
                add(head, source, right_target)

    starts = {state.value for state in graph_automaton.start_states}
    finals = {state.value for state in graph_automaton.final_states}
    start_symbol = grammar.start_symbol

    return {
        (source, target)
        for variable, source, target in triples
        if variable == start_symbol and source in starts and target in finals
    }
