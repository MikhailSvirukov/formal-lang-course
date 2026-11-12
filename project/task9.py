from collections import defaultdict, deque

from networkx import DiGraph
from pyformlang.rsa import RecursiveAutomaton

from project.task2 import graph_to_nfa


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    graph_automaton = graph_to_nfa(graph, start_nodes, final_nodes)
    if rsm.initial_label not in rsm.boxes:
        return set()

    graph_transitions = defaultdict(set)
    for source, symbol, target in graph_automaton:
        graph_transitions[source.value, symbol].add(target.value)

    box_transitions = {}
    box_starts = {}
    box_finals = {}
    for label, box in rsm.boxes.items():
        transitions = defaultdict(list)
        for source, symbol, target in box.dfa:
            transitions[source].append((symbol, target))
        box_transitions[label] = transitions
        box_starts[label] = box.dfa.start_states
        box_finals[label] = box.dfa.final_states

    queue = deque()
    descriptors = set()
    gss_edges = defaultdict(set)
    popped = defaultdict(set)

    def add_descriptor(state, stack_node, vertex):
        descriptor = (state, stack_node, vertex)
        if descriptor not in descriptors:
            descriptors.add(descriptor)
            queue.append(descriptor)

    for source in graph_automaton.start_states:
        stack_node = (rsm.initial_label, source.value)
        for state in box_starts[rsm.initial_label]:
            add_descriptor(state, stack_node, source.value)

    while queue:
        state, stack_node, vertex = queue.popleft()
        label, _ = stack_node

        if state in box_finals[label] and vertex not in popped[stack_node]:
            popped[stack_node].add(vertex)
            for caller, return_state in tuple(gss_edges[stack_node]):
                add_descriptor(return_state, caller, vertex)

        for symbol, target in box_transitions[label].get(state, ()):
            if symbol in rsm.boxes:
                callee = (symbol, vertex)
                continuation = (stack_node, target)
                if continuation not in gss_edges[callee]:
                    gss_edges[callee].add(continuation)
                    for end in tuple(popped[callee]):
                        add_descriptor(target, stack_node, end)
                    for start in box_starts[symbol]:
                        add_descriptor(start, callee, vertex)
            else:
                for end in graph_transitions.get((vertex, symbol), ()):
                    add_descriptor(target, stack_node, end)

    final_vertices = {state.value for state in graph_automaton.final_states}
    return {
        (source.value, target)
        for source in graph_automaton.start_states
        for target in popped.get((rsm.initial_label, source.value), ())
        if target in final_vertices
    }
