from pathlib import Path

import networkx as nx

from project.basic.helpers import create_two_cycles_graph, get_graph_info


def get_labeled_edges(graph: nx.MultiDiGraph) -> set[tuple[object, object, str]]:
    return {
        (source, target, data["label"])
        for source, target, data in graph.edges(data=True)
    }


def test_get_graph_info(monkeypatch, tmp_path: Path):
    graph_file = tmp_path / "graph.csv"
    graph_file.write_text("0 1 a\n1 2 b\n2 0 a\n", encoding="utf-8")
    monkeypatch.setattr("project.basic.helpers.download", lambda _: graph_file)

    assert get_graph_info("example") == (3, 3, {"a", "b"})


def test_create_two_cycles_from_numbers_and_save(tmp_path: Path):
    output_file = tmp_path / "two_cycles.dot"

    graph = create_two_cycles_graph(
        2,
        3,
        labels=("left", "right"),
        output_file=output_file,
    )

    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 7
    assert get_labeled_edges(graph) == {
        (0, 1, "left"),
        (1, 2, "left"),
        (2, 0, "left"),
        (0, 3, "right"),
        (3, 4, "right"),
        (4, 5, "right"),
        (5, 0, "right"),
    }

    restored_graph = nx.drawing.nx_pydot.read_dot(output_file)
    assert restored_graph.number_of_nodes() == graph.number_of_nodes()
    assert restored_graph.number_of_edges() == graph.number_of_edges()


def test_create_two_cycles_from_number_and_iterator(tmp_path: Path):
    output_file = tmp_path / "number_and_iterator.dot"

    graph = create_two_cycles_graph(
        2,
        iter((10, 11, 12)),
        common_node=10,
        labels=("left", "right"),
        output_file=output_file,
    )

    assert output_file.is_file()
    assert set(graph.nodes) == {1, 2, 10, 11, 12}
    assert get_labeled_edges(graph) == {
        (10, 1, "left"),
        (1, 2, "left"),
        (2, 10, "left"),
        (10, 10, "right"),
        (10, 11, "right"),
        (11, 12, "right"),
        (12, 10, "right"),
    }


def test_create_two_cycles_from_iterator_and_number(tmp_path: Path):
    output_file = tmp_path / "iterator_and_number.dot"

    graph = create_two_cycles_graph(
        iter((10, 11)),
        2,
        common_node=10,
        output_file=output_file,
    )

    assert output_file.is_file()
    assert set(graph.nodes) == {3, 4, 10, 11}
    assert get_labeled_edges(graph) == {
        (10, 10, "a"),
        (10, 11, "a"),
        (11, 10, "a"),
        (10, 3, "b"),
        (3, 4, "b"),
        (4, 10, "b"),
    }


def test_create_two_cycles_from_iterators_without_saving():
    graph = create_two_cycles_graph(
        iter(("uno", "dos", "tres")),
        iter((20, 21)),
        common_node=8.9,
    )

    assert set(graph.nodes) == {20, 21, "uno", "dos", "tres", 8.9}
    assert get_labeled_edges(graph) == {
        ("uno", "dos", "a"),
        ("dos", "tres", "a"),
        ("tres", 8.9, "a"),
        (8.9, "uno", "a"),
        (8.9, 20, "b"),
        (20, 21, "b"),
        (21, 8.9, "b"),
    }
