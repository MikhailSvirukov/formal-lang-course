import networkx as nx

from project.task3 import tensor_based_rpq
from project.task4 import ms_bfs_based_rpq


def test_ms_bfs_finds_long_paths_from_several_sources():
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

    result = ms_bfs_based_rpq(
        "open read retry read (retry read)* transform close",
        graph,
        {0, 10, 20},
        {4, 14, 24},
    )

    assert result == {(0, 4), (10, 14)}


def test_ms_bfs_keeps_results_for_each_source_separate():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 3, label="c")
    graph.add_edge(10, 1, label="a")
    graph.add_edge(20, 2, label="b")
    graph.add_edge(3, 1, label="loop")

    assert ms_bfs_based_rpq(
        "a b c (loop b c)*",
        graph,
        {0, 10, 20},
        {2, 3},
    ) == {(0, 3), (10, 3)}


def test_ms_bfs_includes_zero_length_paths_only_for_allowed_pairs():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="a")

    assert ms_bfs_based_rpq("(a b)*", graph, {0, 1, 2}, {0, 2}) == {
        (0, 0),
        (0, 2),
        (2, 2),
    }


def test_ms_bfs_matches_tensor_algorithm_on_branching_graph():
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
            (10, 2, {"label": "a"}),
        ]
    )
    regex = "a (b|c) (d (b|c))* b c"
    starts = {0, 10}
    finals = {3, 4, 5}

    assert ms_bfs_based_rpq(regex, graph, starts, finals) == tensor_based_rpq(
        regex, graph, starts, finals
    )


def test_ms_bfs_does_not_modify_inputs():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="b")
    starts = {0}
    finals = {1}
    graph_before = graph.copy()

    ms_bfs_based_rpq("(a b)* a", graph, starts, finals)

    assert nx.utils.graphs_equal(graph, graph_before)
    assert starts == {0}
    assert finals == {1}
