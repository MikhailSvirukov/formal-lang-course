from pathlib import Path

import networkx as nx
from cfpq_data import graph_from_csv

from project.basic.helpers import create_two_cycles_graph, get_graph_info
from project.task2 import graph_to_nfa, regex_to_dfa


def test_regex_to_dfa_is_minimal_and_preserves_language():
    dfa = regex_to_dfa("(a|b)* a b")

    assert dfa.is_deterministic()
    assert len(dfa.states) == len(dfa.minimize().states)
    assert dfa.accepts(["a", "b"])
    assert dfa.accepts(["b", "a", "a", "b"])
    assert not dfa.accepts(["a", "a"])


def test_graph_to_nfa_uses_selected_start_and_final_states():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 1, label="c")
    graph.add_edge(2, 3, label="d")
    graph.add_edge(0, 4, label="x")
    graph.add_edge(4, 5, label="y")
    graph.add_edge(5, 2, label="z")

    nfa = graph_to_nfa(graph, {0}, {3})

    assert nfa.accepts(["a", "b", "c", "b", "c", "b", "d"])
    assert nfa.accepts(["x", "y", "z", "c", "b", "d"])
    assert not nfa.accepts(["a", "b", "c", "b"])
    assert not nfa.accepts(["x", "y", "z", "b", "d"])


def test_graph_to_nfa_preserves_parallel_edges():
    graph = nx.MultiDiGraph()
    graph.add_edge("source", "left", label="a")
    graph.add_edge("source", "left", label="b")
    graph.add_edge("left", "middle", label="c")
    graph.add_edge("middle", "left", label="d")
    graph.add_edge("middle", "target", label="e")

    nfa = graph_to_nfa(graph, {"source"}, {"target"})

    assert nfa.accepts(["a", "c", "d", "c", "d", "c", "e"])
    assert nfa.accepts(["b", "c", "d", "c", "e"])
    assert not nfa.accepts(["a", "d", "c", "e"])


def test_empty_state_sets_make_every_vertex_start_and_final():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="c")

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.accepts([])
    assert nfa.accepts(["a", "b", "c", "a", "b", "c", "a"])
    assert nfa.accepts(["b", "c", "a", "b", "c"])
    assert not nfa.accepts(["a", "c", "b"])
    assert {state.value for state in nfa.states} == {0, 1, 2}


def test_graph_to_nfa_does_not_modify_arguments():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    starts = {0}
    finals = {1}
    graph_before = graph.copy()

    graph_to_nfa(graph, starts, finals)

    assert nx.utils.graphs_equal(graph, graph_before)
    assert starts == {0}
    assert finals == {1}


def test_graph_to_nfa_with_two_cycles_graph_from_task_1():
    graph = create_two_cycles_graph(
        2,
        3,
        labels=("left", "right"),
    )

    nfa = graph_to_nfa(graph, {0}, {0})

    assert nfa.accepts(["left"] * 3 + ["right"] * 4)
    assert nfa.accepts(["right"] * 8 + ["left"] * 6)
    assert not nfa.accepts(["left"] * 2 + ["right"] * 4)
    assert not nfa.accepts(["left", "right"])


def test_graph_to_nfa_with_non_integer_vertices_from_task_1():
    graph = create_two_cycles_graph(
        iter(("first", "second")),
        iter((10, 11)),
        common_node=("common", 0),
        labels=("a", "b"),
    )

    nfa = graph_to_nfa(graph, {("common", 0)}, {("common", 0)})

    assert nfa.accepts(["a"] * 6 + ["b"] * 9 + ["a"] * 3)
    assert nfa.accepts(["b"] * 6 + ["a"] * 9)
    assert not nfa.accepts(["a"] * 4 + ["b"] * 3)


def test_graph_to_nfa_with_cfpq_data_graph_used_by_task_1(monkeypatch, tmp_path: Path):
    graph_file = tmp_path / "graph.csv"
    graph_file.write_text(
        "0 1 open\n1 2 read\n2 1 retry\n2 3 transform\n3 4 close\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("project.basic.helpers.download", lambda _: graph_file)

    assert get_graph_info("example") == (
        5,
        5,
        {"open", "read", "retry", "transform", "close"},
    )
    graph = graph_from_csv(graph_file)
    nfa = graph_to_nfa(graph, {0}, {4})

    assert isinstance(graph, nx.MultiDiGraph)
    assert nfa.accepts(["open", "read", "retry", "read", "transform", "close"])
    assert nfa.accepts(
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
    assert not nfa.accepts(["open", "read", "retry", "transform", "close"])
