import networkx as nx
from pyformlang.cfg import CFG

from project.task6 import cfg_to_weak_normal_form, hellings_based_cfpq
from project.task8 import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq


def test_cfg_to_rsm_builds_a_box_for_each_variable():
    grammar = CFG.from_text(
        """
        S -> $ | a S b | Pair Pair
        Pair -> c d
        """
    )

    rsm = cfg_to_rsm(grammar)
    normal_form = cfg_to_weak_normal_form(grammar)

    assert rsm.initial_label.value == "S"
    assert {label.value for label in rsm.labels} == {
        variable.value for variable in normal_form.variables
    }
    assert all(rsm.get_box(label) is not None for label in rsm.labels)


def test_ebnf_to_rsm_preserves_regular_bodies():
    rsm = ebnf_to_rsm(
        """
        S -> open Body close
        Body -> (read | retry)* transform
        """
    )

    assert rsm.get_box("S").dfa.accepts(["open", "Body", "close"])
    assert rsm.get_box("Body").dfa.accepts(["transform"])
    assert rsm.get_box("Body").dfa.accepts(
        ["read", "retry", "read", "transform"]
    )


def test_tensor_algorithm_finds_recursive_and_concatenated_paths():
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (1, 2, {"label": "a"}),
            (2, 3, {"label": "b"}),
            (3, 4, {"label": "b"}),
            (4, 5, {"label": "a"}),
            (5, 6, {"label": "b"}),
        ]
    )
    grammar = CFG.from_text("S -> $ | a S b | S S")

    assert tensor_based_cfpq(cfg_to_rsm(grammar), graph, {0, 1}, {3, 4, 6}) == {
        (0, 4),
        (0, 6),
        (1, 3),
    }


def test_tensor_algorithm_handles_ebnf_and_cycles():
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "open"}),
            (1, 2, {"label": "read"}),
            (2, 1, {"label": "retry"}),
            (2, 3, {"label": "transform"}),
            (3, 4, {"label": "close"}),
        ]
    )
    rsm = ebnf_to_rsm(
        """
        S -> open Body close
        Body -> read (retry read)* transform
        """
    )

    assert tensor_based_cfpq(rsm, graph, {0}, {4}) == {(0, 4)}


def test_tensor_algorithm_does_not_treat_box_named_edge_as_a_call():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="A")
    graph.add_edge(20, 30, label="b")
    rsm = ebnf_to_rsm("Root -> A b\nA -> a")

    assert rsm.initial_label.value == "Root"
    assert tensor_based_cfpq(rsm, graph) == set()

    graph.add_edge(10, 20, label="a")
    assert tensor_based_cfpq(rsm, graph) == {(10, 30)}


def test_tensor_algorithm_matches_hellings_on_branching_graph():
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

    assert tensor_based_cfpq(
        cfg_to_rsm(grammar), graph, starts, finals
    ) == hellings_based_cfpq(grammar, graph, starts, finals)
