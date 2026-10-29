import networkx as nx
from pyformlang.cfg import CFG

from project.task6 import hellings_based_cfpq
from project.task7 import matrix_based_cfpq


def test_matrix_algorithm_finds_nested_and_concatenated_paths():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="b")
    graph.add_edge(4, 5, label="a")
    graph.add_edge(5, 6, label="b")
    grammar = CFG.from_text("S -> $ | a S b | S S")

    assert matrix_based_cfpq(grammar, graph, {0, 1}, {3, 4, 6}) == {
        (0, 4),
        (0, 6),
        (1, 3),
    }


def test_matrix_algorithm_reaches_fixed_point_on_cycle():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 0, label="a")
    grammar = CFG.from_text("S -> a | S S")

    assert matrix_based_cfpq(grammar, graph) == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }


def test_matrix_algorithm_handles_long_and_unit_productions():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 11, label="open")
    graph.add_edge(11, 12, label="read")
    graph.add_edge(12, 11, label="retry")
    graph.add_edge(12, 13, label="transform")
    graph.add_edge(13, 14, label="close")
    grammar = CFG.from_text(
        """
        S -> Request
        Request -> open Body close
        Body -> read transform | read retry Body
        """
    )

    assert matrix_based_cfpq(grammar, graph, {10}, {14}) == {(10, 14)}


def test_matrix_algorithm_includes_empty_paths():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])

    assert matrix_based_cfpq(CFG.from_text("S -> $"), graph) == {
        (0, 0),
        (1, 1),
        (2, 2),
    }


def test_matrix_algorithm_matches_hellings_on_branching_graph():
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
    grammar = CFG.from_text(
        """
        S -> a B Tail
        B -> b | c
        Tail -> b c | d B Tail
        """
    )
    starts = {0, 2}
    finals = {3, 4, 5}

    assert matrix_based_cfpq(grammar, graph, starts, finals) == hellings_based_cfpq(
        grammar, graph, starts, finals
    )
