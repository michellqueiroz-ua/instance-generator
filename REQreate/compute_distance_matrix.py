import math
from collections import OrderedDict

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .logger_utils import get_logger


class DummyRay:
    """Stand-in for ray so the unguarded ray calls below become no-ops.

    Only installed when ray is genuinely missing. It is not a parallel
    executor: ``put``/``get`` are the identity and ``remote`` runs the
    function inline, so every caller must still take the sequential branch
    for the actual work.
    """
    @staticmethod
    def remote(func):
        class RemoteWrapper:
            def remote(*args, **kwargs):
                return func(*args, **kwargs)
        return RemoteWrapper
    @staticmethod
    def shutdown():
        pass
    @staticmethod
    def init(*args, **kwargs):
        pass
    @staticmethod
    def put(obj):
        return obj
    @staticmethod
    def get(objs):
        return objs


try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = DummyRay()
    print("Warning: Ray not available. Using sequential processing (slower but works on Windows).")


def _ray_init():
    """Restart a local ray instance, sized by ray from the host machine.

    A no-op when ray is missing, so call sites can call it unconditionally.

    The previous call sites hardcoded ``num_cpus=8`` and a 14 GB object store
    (and one of them disagreed with the other three), which fails outright on
    any machine with less RAM or fewer cores. Passing nothing lets ray read
    the actual core count and available memory, so there is one helper and no
    fixed sizes to outgrow.
    """
    if not RAY_AVAILABLE:
        return
    ray.shutdown()
    ray.init()


def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Non-ray version for sequential processing
def shortest_path_nx_sequential(G, u, v, weight='length'):
    try:
        shortest_path_length = nx.dijkstra_path_length(G, u, v, weight=weight)
        return shortest_path_length
    except nx.NetworkXNoPath:
        return -1

# Non-ray version for single source to all nodes
def shortest_path_nx_ss_sequential(G, u, weight):
    shortest_path_length_u = {}
    shortest_path_length_u = nx.single_source_dijkstra_path_length(G, u, weight=weight)
    return shortest_path_length_u


if RAY_AVAILABLE:
    @ray.remote
    def shortest_path_nx(G, u, v):
        try:
            shortest_path_length = nx.dijkstra_path_length(G, u, v, weight='length')
            return shortest_path_length
        except nx.NetworkXNoPath:
            return -1

    #ss -> single source to all nodes
    @ray.remote
    def shortest_path_nx_ss(G, u, weight):
        shortest_path_length_u = {}
        shortest_path_length_u = nx.single_source_dijkstra_path_length(G, u, weight=weight)
        return shortest_path_length_u


# Rows kept in memory per matrix. A row is one float32/int32 per graph node,
# so 5000 rows of a 26k-node city take about 500 MB.
MAX_CACHED_ROWS = 5000

_UNREACHABLE_INT = -1


def _node_key(node):
    """Accept node ids as ints, numpy ints, or the old string column labels."""
    if isinstance(node, str):
        try:
            return int(node)
        except ValueError:
            return int(float(node))
    return int(node)


class _Loc:
    def __init__(self, matrix):
        self._matrix = matrix

    def __getitem__(self, key):
        origin, destination = key
        return self._matrix.get(origin, destination)


class LazyDistanceMatrix:
    """Shortest-path matrix over a graph, computed one origin row at a time.

    Replaces the all-pairs DataFrame, which needed ~75 GB for a 26k-node
    city. Lookups keep the old DataFrame syntax, ``matrix.loc[origin, "dest"]``:
    the first lookup from an origin runs one Dijkstra over the graph (tens of
    milliseconds) and caches that row; later lookups are array reads.

    Unreachable destinations return NaN and unknown nodes raise KeyError,
    as the DataFrame did.

    Travel times (``value_type=int``) are truncated to whole seconds in
    float64 before being stored as int32, so they match the old matrix
    exactly. Distances are stored as float32 (sub-millimetre differences).
    """

    def __init__(self, G, weight, value_type=float, max_rows=MAX_CACHED_ROWS):
        self.weight = weight
        self.value_type = value_type
        self.max_rows = max_rows
        self.directed = G.is_directed()
        self.nodes = list(G.nodes)
        self.node_to_index = {node: index for index, node in enumerate(self.nodes)}

        # Parallel edges collapse to the cheapest one.
        edge_weights = {}
        for u, v, attributes in G.edges(data=True):
            edge = (self.node_to_index[u], self.node_to_index[v])
            edge_weight = float(attributes.get(weight, 1))
            if edge not in edge_weights or edge_weight < edge_weights[edge]:
                edge_weights[edge] = edge_weight
        edges = list(edge_weights)
        self.graph = csr_matrix(
            (
                [edge_weights[edge] for edge in edges],
                ([edge[0] for edge in edges], [edge[1] for edge in edges]),
            ),
            shape=(len(self.nodes), len(self.nodes)),
        )
        self._rows = OrderedDict()
        self.loc = _Loc(self)

    def __contains__(self, node):
        try:
            return _node_key(node) in self.node_to_index
        except (TypeError, ValueError):
            return False

    def _index(self, node):
        return self.node_to_index[_node_key(node)]

    def row(self, origin):
        index = self._index(origin)
        cached = self._rows.get(index)
        if cached is not None:
            self._rows.move_to_end(index)
            return cached

        distances = dijkstra(self.graph, directed=self.directed, indices=index)
        if self.value_type is int:
            reachable = np.isfinite(distances)
            values = np.full(distances.shape, _UNREACHABLE_INT, dtype=np.int32)
            values[reachable] = np.trunc(distances[reachable]).astype(np.int32)
        else:
            values = distances.astype(np.float32)

        self._rows[index] = values
        if len(self._rows) > self.max_rows:
            self._rows.popitem(last=False)
        return values

    def get(self, origin, destination):
        destination_index = self._index(destination)
        value = self.row(origin)[destination_index]
        if self.value_type is int:
            return math.nan if value == _UNREACHABLE_INT else int(value)
        return float(value) if np.isfinite(value) else math.nan

    def __getstate__(self):
        # The network object is pickled; ship the graph, not the cache.
        state = self.__dict__.copy()
        state["_rows"] = OrderedDict()
        del state["loc"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.loc = _Loc(self)

    def __getattr__(self, name):
        # Fail loudly if code still expects the whole DataFrame.
        if name in ("index", "columns", "iterrows", "to_csv", "values", "shape"):
            raise AttributeError(
                f"LazyDistanceMatrix has no '{name}': the full matrix is no "
                "longer materialised; look up values with .loc[origin, dest]."
            )
        raise AttributeError(name)


def travel_times_between(G, nodes, weight="travel_time", chunk=64, progress=None):
    """Whole-second shortest-path times between ``nodes`` (one row and column per entry).

    Same values as ``LazyDistanceMatrix(G, weight, int).get``: edge weights are summed in
    float64 and truncated to whole seconds. Instead of caching one full-graph row per origin
    (about 4 MB per origin on a 1M-node graph, so 3500 stations need ~14 GB), the searches run
    ``chunk`` origins at a time and only the columns of ``nodes`` are kept.

    Returns an int32 array; unreachable pairs are -1.
    """
    lazy = LazyDistanceMatrix(G, weight, int)
    unique = list(dict.fromkeys(_node_key(n) for n in nodes))
    position = {node: i for i, node in enumerate(unique)}
    columns = np.array([lazy.node_to_index[node] for node in unique])
    small = np.full((len(unique), len(unique)), _UNREACHABLE_INT, dtype=np.int32)
    for start in range(0, len(unique), chunk):
        indices = columns[start:start + chunk]
        rows = dijkstra(lazy.graph, directed=lazy.directed, indices=indices)[:, columns]
        reachable = np.isfinite(rows)
        block = np.full(rows.shape, _UNREACHABLE_INT, dtype=np.int32)
        block[reachable] = np.trunc(rows[reachable]).astype(np.int32)
        small[start:start + len(indices)] = block
        if progress:
            progress(min(start + chunk, len(unique)), len(unique))
    order = np.array([position[_node_key(n)] for n in nodes])
    return small[np.ix_(order, order)]


def _update_distance_matrix_walk(G_walk, bus_stops_fr, save_dir, output_file_base):
    # The lazy matrix answers any origin, so extra stops need no update.
    return LazyDistanceMatrix(G_walk, "length")


def _get_distance_matrix(G_walk, G_drive, bus_stops, save_dir, output_file_base):
    logger = get_logger()
    if logger:
        logger.info('Preparing on-demand distance matrices (computed per origin as needed)')
    else:
        print('Preparing on-demand distance matrices...')

    shortest_path_walk = LazyDistanceMatrix(G_walk, "length")
    shortest_path_drive = LazyDistanceMatrix(G_drive, "travel_time", int)
    shortest_dist_drive = LazyDistanceMatrix(G_drive, "length")

    # Callers ignore this list; it was derived from the full matrix.
    unreachable_nodes = []

    return shortest_path_walk, shortest_path_drive, shortest_dist_drive, unreachable_nodes
