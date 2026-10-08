import networkx as nx

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import (
    AdjacencyMatrixFA,
    intersect_automata,
    tensor_based_rpq,
)


def test_adjacency_matrix_fa_accepts_long_words_with_cycles():
    automaton = AdjacencyMatrixFA(regex_to_dfa("a b (c b)* d"))

    assert automaton.accepts(["a", "b", "c", "b", "c", "b", "d"])
    assert automaton.accepts(["a", "b"] + ["c", "b"] * 12 + ["d"])
    assert not automaton.accepts(["a", "b", "c", "c", "b", "d"])
    assert not automaton.accepts(["a", "b"] + ["c", "b"] * 5)


def test_adjacency_matrix_fa_handles_nondeterministic_long_paths():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="open")
    graph.add_edge(0, 1, label="skip")
    graph.add_edge(1, 2, label="read")
    graph.add_edge(2, 1, label="retry")
    graph.add_edge(2, 3, label="transform")
    graph.add_edge(3, 4, label="close")
    automaton = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {4}))

    assert automaton.accepts(
        [
            "open",
            "read",
            "retry",
            "read",
            "retry",
            "read",
            "transform",
            "close",
        ]
    )
    assert automaton.accepts(["skip", "read", "retry", "read", "transform", "close"])
    assert not automaton.accepts(["open", "read", "retry", "transform", "close"])


def test_is_empty_accounts_for_reachability_and_empty_word():
    unreachable = nx.MultiDiGraph()
    unreachable.add_edge(0, 1, label="a")
    unreachable.add_edge(2, 3, label="b")
    empty_language = AdjacencyMatrixFA(graph_to_nfa(unreachable, {0}, {3}))
    epsilon_language = AdjacencyMatrixFA(regex_to_dfa("(a b)*"))

    assert empty_language.is_empty()
    assert not epsilon_language.is_empty()
    assert epsilon_language.accepts([])


def test_intersection_accepts_only_words_shared_by_both_automata():
    first = AdjacencyMatrixFA(regex_to_dfa("a (b|c)* d"))
    second = AdjacencyMatrixFA(regex_to_dfa("a b* d"))
    intersection = intersect_automata(first, second)

    assert intersection.accepts(["a"] + ["b"] * 15 + ["d"])
    assert intersection.accepts(["a", "d"])
    assert not intersection.accepts(["a", "b", "c", "b", "d"])
    assert not intersection.accepts(["a"] + ["b"] * 10 + ["c", "d"])
    assert not intersection.is_empty()


def test_intersection_detects_disjoint_languages():
    first = AdjacencyMatrixFA(regex_to_dfa("a b* c"))
    second = AdjacencyMatrixFA(regex_to_dfa("a b* d"))

    assert intersect_automata(first, second).is_empty()


def test_tensor_based_rpq_finds_pairs_connected_by_long_matching_paths():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="open")
    graph.add_edge(1, 2, label="read")
    graph.add_edge(2, 1, label="retry")
    graph.add_edge(2, 3, label="transform")
    graph.add_edge(3, 4, label="close")

    graph.add_edge(10, 11, label="open")
    graph.add_edge(11, 12, label="read")
    graph.add_edge(12, 11, label="retry")
    graph.add_edge(12, 13, label="transform")
    graph.add_edge(13, 14, label="close")

    graph.add_edge(20, 21, label="open")
    graph.add_edge(21, 22, label="read")
    graph.add_edge(22, 23, label="transform")
    graph.add_edge(23, 24, label="close")

    result = tensor_based_rpq(
        "open read retry read (retry read)* transform close",
        graph,
        {0, 10, 20},
        {4, 14, 24},
    )

    assert result == {(0, 4), (10, 14)}


def test_tensor_based_rpq_includes_zero_length_paths():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="a")

    assert tensor_based_rpq("(a b)*", graph, {0, 1, 2}, {0, 1, 2}) == {
        (0, 0),
        (1, 1),
        (2, 2),
        (0, 2),
    }
