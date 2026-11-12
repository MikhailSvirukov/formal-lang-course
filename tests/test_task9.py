import networkx as nx

from project.task8 import ebnf_to_rsm, tensor_based_cfpq
from project.task9 import gll_based_cfpq


def test_gll_reuses_nullable_call_for_late_caller():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 10, label="c")
    graph.add_edge(10, 20, label="b")
    graph.add_edge(10, 30, label="d")
    rsm = ebnf_to_rsm("S -> A b | c A d\nA -> $")

    assert gll_based_cfpq(rsm, graph, {10}, {20, 30}) == {
        (10, 20),
        (10, 30),
    }


def test_gll_shared_call_has_multiple_continuations():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    graph.add_edge(20, 40, label="c")
    rsm = ebnf_to_rsm("S -> A b | A c\nA -> a")

    assert gll_based_cfpq(rsm, graph, {10}, {30, 40}) == {
        (10, 30),
        (10, 40),
    }


def test_gll_handles_left_recursion_and_nullable_paths():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="a")
    graph.add_node(2)
    rsm = ebnf_to_rsm("S -> S a | $")

    assert gll_based_cfpq(rsm, graph) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 2),
    }


def test_gll_finds_nested_paths_through_unselected_internal_vertices():
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        (vertex, vertex + 1, {"label": label}) for vertex, label in enumerate("aabb")
    )
    rsm = ebnf_to_rsm("S -> a S b | $")

    assert gll_based_cfpq(rsm, graph, {0, 1}, {3, 4}) == {
        (0, 4),
        (1, 3),
    }


def test_gll_matches_tensor_algorithm_on_recursive_branching_graph():
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (0, 2, {"label": "a"}),
            (1, 3, {"label": "b"}),
            (2, 3, {"label": "c"}),
            (3, 1, {"label": "d"}),
            (3, 4, {"label": "b"}),
            (4, 5, {"label": "c"}),
            (5, 3, {"label": "d"}),
        ]
    )
    rsm = ebnf_to_rsm(
        """
        S -> a B Tail
        B -> b | c
        Tail -> b c | d B Tail
        """
    )
    starts = {0, 2}
    finals = {3, 4, 5}

    assert gll_based_cfpq(rsm, graph, starts, finals) == tensor_based_cfpq(
        rsm, graph, starts, finals
    )
