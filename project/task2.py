"""Construction of finite automata from regular expressions and graphs."""

from collections.abc import Set
from typing import Hashable

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: Set[Hashable] | None = None,
    final_states: Set[Hashable] | None = None,
) -> NondeterministicFiniteAutomaton:
    nodes = set(graph.nodes)
    actual_start_states = nodes if not start_states else set(start_states)
    actual_final_states = nodes if not final_states else set(final_states)

    nfa = NondeterministicFiniteAutomaton(
        states=nodes,
        start_state=actual_start_states,
        final_states=actual_final_states,
    )

    for source, target, data in graph.edges(data=True):
        nfa.add_transition(source, data["label"], target)

    return nfa
