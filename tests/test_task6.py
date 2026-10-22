import networkx as nx
from pyformlang.cfg import CFG, Terminal, Variable

from project.task6 import cfg_to_weak_normal_form, hellings_based_cfpq


def test_weak_normal_form_has_only_allowed_productions():
    grammar = CFG.from_text(
        """
        S -> $ | a S b | S S | C
        C -> c d e
        """
    )

    normal_form = cfg_to_weak_normal_form(grammar)

    assert normal_form.generate_epsilon()
    for production in normal_form.productions:
        body = production.body
        assert (
            len(body) == 0
            or (len(body) == 1 and isinstance(body[0], Terminal))
            or (
                len(body) == 2
                and isinstance(body[0], Variable)
                and isinstance(body[1], Variable)
            )
        )


def test_weak_normal_form_preserves_words_and_empty_word():
    grammar = CFG.from_text("S -> $ | a S b | S S")
    normal_form = cfg_to_weak_normal_form(grammar)
    words = [
        [],
        ["a", "b"],
        ["a", "a", "b", "b"],
        ["a", "b", "a", "b"],
        ["a", "a", "b"],
        ["b", "a"],
    ]

    for word in words:
        assert grammar.contains(word) == normal_form.contains(word)


def test_hellings_finds_nested_and_concatenated_paths():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="b")
    graph.add_edge(4, 5, label="a")
    graph.add_edge(5, 6, label="b")
    grammar = CFG.from_text("S -> $ | a S b | S S")

    assert hellings_based_cfpq(grammar, graph, {0, 1}, {3, 4, 6}) == {
        (0, 4),
        (0, 6),
        (1, 3),
    }


def test_hellings_handles_long_productions_and_unit_rules():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 11, label="open")
    graph.add_edge(11, 12, label="read")
    graph.add_edge(12, 13, label="transform")
    graph.add_edge(13, 14, label="close")
    graph.add_edge(12, 11, label="retry")
    grammar = CFG.from_text(
        """
        S -> Request
        Request -> open Body close
        Body -> read transform | read retry Body
        """
    )

    assert hellings_based_cfpq(grammar, graph, {10}, {14}) == {(10, 14)}


def test_hellings_returns_empty_paths_on_isolated_vertices():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    grammar = CFG.from_text("S -> $")

    assert hellings_based_cfpq(grammar, graph) == {(0, 0), (1, 1), (2, 2)}


def test_hellings_keeps_parallel_labels_and_filters_endpoints():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")
    graph.add_edge(1, 2, label="c")
    graph.add_edge(2, 3, label="d")
    grammar = CFG.from_text("S -> A C | B C\nA -> a\nB -> b\nC -> c d")

    assert hellings_based_cfpq(grammar, graph, {0, 1}, {2, 3}) == {(0, 3)}
