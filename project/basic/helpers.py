from os import PathLike
from typing import TypeAlias, Iterable, Any

from cfpq_data import download, graph_from_csv, labeled_two_cycles_graph
from networkx import MultiDiGraph
from networkx.drawing.nx_pydot import write_dot

GraphInfo: TypeAlias = tuple[int, int, set[str]]


def get_unique_labels(
    graph: MultiDiGraph,
) -> set[str]:
    labels = set()
    for _, _, data in graph.edges(data=True):
        label = data.get("label")
        if label is not None:
            labels.add(label)
    return labels


def get_graph_info(graph_name: str) -> GraphInfo:
    graph = graph_from_csv(download(graph_name))
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    labels = get_unique_labels(graph)
    return nodes, edges, labels


def create_two_cycles_graph(
    first_cycle_size: int | Iterable[Any],
    second_cycle_size: int | Iterable[Any],
    common_node: int | Any = 0,
    labels: tuple[str, str] = ("a", "b"),
    output_file: str | PathLike[str] | None = None,
) -> MultiDiGraph:
    graph = labeled_two_cycles_graph(
        first_cycle_size,
        second_cycle_size,
        common_node=common_node,
        labels=labels,
    )
    if output_file is not None:
        write_dot(graph, output_file)
    return graph
