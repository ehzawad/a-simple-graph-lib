from typing import Any, Dict, Set, Optional, Iterator, List, Tuple
from collections import defaultdict
import json

class GraphException(Exception):
    """Base exception class for Graph operations."""
    pass

class VertexNotFoundError(GraphException):
    """Raised when a vertex is not found in the graph."""
    pass

class EdgeNotFoundError(GraphException):
    """Raised when an edge is not found in the graph."""
    pass

class InvalidOperationError(GraphException):
    """Raised when an invalid operation is attempted on the graph."""
    pass

class Vertex:
    def __init__(self, value: Any):
        self.value = value
        self.edges: Set['Edge'] = set()

    @property
    def degree(self) -> int:
        return len(self.edges)

    @property
    def in_degree(self) -> int:
        return sum(1 for edge in self.edges if edge.target == self)

    @property
    def out_degree(self) -> int:
        return sum(1 for edge in self.edges if edge.source == self)

    @property
    def is_leaf(self) -> bool:
        if all(edge.directed for edge in self.edges):
            return self.in_degree > 0 and self.out_degree == 0
        return self.degree == 1

    @property
    def is_root(self) -> bool:
        return all(edge.directed for edge in self.edges) and self.in_degree == 0 and self.out_degree > 0

    @property
    def has_edges(self) -> bool:
        return self.degree > 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vertex):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"Vertex({self.value}, degree={self.degree})"

    def __lt__(self, other: 'Vertex') -> bool:
        return self.value < other.value

class Edge:
    def __init__(self, source: Vertex, target: Vertex, weight: float = 1.0, directed: bool = False):
        self.source = source
        self.target = target
        self.weight = weight
        self.directed = directed

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        if self.directed:
            return (self.source, self.target, self.weight, self.directed) == (other.source, other.target, other.weight, other.directed)
        else:
            return ((self.source, self.target) == (other.source, other.target) or
                    (self.source, self.target) == (other.target, other.source)) and self.weight == other.weight and self.directed == other.directed

    def __hash__(self) -> int:
        if self.directed:
            return hash((self.source, self.target, self.weight, self.directed))
        else:
            return hash(frozenset([self.source, self.target]) | frozenset([self.weight, self.directed]))

    def __repr__(self) -> str:
        return f"Edge({self.source.value} {'-->' if self.directed else '<->'} {self.target.value}, weight={self.weight})"

class Graph:
    def __init__(self, directed: bool = False):
        self._vertices: Dict[Any, Vertex] = {}
        self._edges: Set[Edge] = set()
        self.directed = directed

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_vertex(self, value: Any) -> Vertex:
        if value not in self._vertices:
            self._vertices[value] = Vertex(value)
        return self._vertices[value]

    def remove_vertex(self, value: Any) -> None:
        if value not in self._vertices:
            raise VertexNotFoundError(f"Vertex not found: {value}")
        
        vertex = self._vertices[value]
        edges_to_remove = vertex.edges.copy()
        
        for edge in edges_to_remove:
            self.remove_edge(edge.source.value, edge.target.value)
        
        del self._vertices[value]

    def add_edge(self, source_value: Any, target_value: Any, weight: float = 1.0, directed: Optional[bool] = None) -> Edge:
        source = self.add_vertex(source_value)
        target = self.add_vertex(target_value)
        if directed is None:
            directed = self.directed
        
        for existing_edge in self._edges:
            if existing_edge.source == source and existing_edge.target == target and existing_edge.directed == directed:
                existing_edge.weight = weight  # Update weight if edge already exists
                return existing_edge
        
        edge = Edge(source, target, weight, directed)
        self._edges.add(edge)
        source.edges.add(edge)
        if source != target or not directed:  # Add to target unless it's a directed self-loop
            target.edges.add(edge)
        
        return edge

    def remove_edge(self, source_value: Any, target_value: Any) -> None:
        source = self._vertices.get(source_value)
        target = self._vertices.get(target_value)
        if not source or not target:
            raise EdgeNotFoundError(f"Edge not found between {source_value} and {target_value}")
        
        edge_to_remove = None
        for edge in self._edges:
            if edge.source == source and edge.target == target:
                edge_to_remove = edge
                break
            if not self.directed and edge.source == target and edge.target == source:
                edge_to_remove = edge
                break
        
        if not edge_to_remove:
            raise EdgeNotFoundError(f"No edge found between {source_value} and {target_value}")
        
        self._edges.remove(edge_to_remove)
        source.edges.remove(edge_to_remove)
        if source != target or not edge_to_remove.directed:  # Remove from target unless it's a directed self-loop
            target.edges.remove(edge_to_remove)

    def get_vertex(self, value: Any) -> Optional[Vertex]:
        return self._vertices.get(value)

    def get_edge(self, source_value: Any, target_value: Any) -> Optional[Edge]:
        source = self._vertices.get(source_value)
        target = self._vertices.get(target_value)
        if source and target:
            for edge in self._edges:
                if edge.source == source and edge.target == target:
                    return edge
                if not self.directed and not edge.directed and edge.source == target and edge.target == source:
                    return edge
        return None

    def are_adjacent(self, value1: Any, value2: Any) -> bool:
        v1, v2 = self.get_vertex(value1), self.get_vertex(value2)
        if v1 and v2:
            for edge in v1.edges:
                if (edge.target == v2) or (not edge.directed and edge.source == v2):
                    return True
            if not self.directed:
                for edge in v2.edges:
                    if (edge.target == v1) or (not edge.directed and edge.source == v1):
                        return True
        return False

    def get_neighbors(self, value: Any) -> Set[Any]:
        vertex = self.get_vertex(value)
        if not vertex:
            return set()
        neighbors = set()
        for edge in vertex.edges:
            if edge.source == vertex:
                neighbors.add(edge.target.value)
            elif not edge.directed:
                neighbors.add(edge.source.value)
        return neighbors

    def get_degree(self, value: Any) -> int:
        vertex = self.get_vertex(value)
        return vertex.degree if vertex else 0

    def is_tree(self) -> bool:
        if not self._vertices:
            return True  # An empty graph is considered a tree
        
        if self.edge_count != self.vertex_count - 1:
            return False
        
        visited = set()
        stack: List[Tuple[Vertex, Optional[Vertex]]] = [(next(iter(self._vertices.values())), None)]
        
        while stack:
            vertex, parent = stack.pop()
            if vertex in visited:
                return False
            visited.add(vertex)
            
            for edge in vertex.edges:
                neighbor = edge.target if edge.source == vertex else edge.source
                if neighbor != parent:
                    stack.append((neighbor, vertex))
        
        return len(visited) == self.vertex_count

    def has_cycle(self) -> bool:
        if self.directed:
            return self._has_cycle_directed()
        else:
            return self._has_cycle_undirected()

    def _has_cycle_directed(self) -> bool:
        visited = set()
        rec_stack = set()

        def dfs(vertex):
            visited.add(vertex)
            rec_stack.add(vertex)

            for edge in vertex.edges:
                if edge.source != vertex:
                    continue
                neighbor = edge.target
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(vertex)
            return False

        for vertex in self._vertices.values():
            if vertex not in visited:
                if dfs(vertex):
                    return True

        return False

    def _has_cycle_undirected(self) -> bool:
        visited = set()

        def dfs(vertex, parent):
            visited.add(vertex)

            for edge in vertex.edges:
                neighbor = edge.target if edge.source == vertex else edge.source
                if neighbor not in visited:
                    if dfs(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    return True

            return False

        for vertex in self._vertices.values():
            if vertex not in visited:
                if dfs(vertex, None):
                    return True

        return False

    def strongly_connected_components(self) -> List[List[Any]]:
        index = 0
        stack = []
        indices = {}
        lowlinks = {}
        on_stack = set()
        components = []

        def strongconnect(v):
            nonlocal index
            indices[v] = index
            lowlinks[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in self.get_neighbors(v):
                if self.get_edge(v, w):  # Ensure there's a directed edge from v to w
                    if w not in indices:
                        strongconnect(w)
                        lowlinks[v] = min(lowlinks[v], lowlinks[w])
                    elif w in on_stack:
                        lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == v:
                        break
                components.append(component)

        for v in self._vertices:
            if v not in indices:
                strongconnect(v)

        return components

    def __len__(self) -> int:
        return self.vertex_count

    def __iter__(self) -> Iterator[Any]:
        return iter(self._vertices)

    def __contains__(self, value: Any) -> bool:
        return value in self._vertices

    def __repr__(self) -> str:
        return f"Graph(vertices={self.vertex_count}, edges={self.edge_count}, directed={self.directed})"

    def is_empty(self) -> bool:
        return self.vertex_count == 0

    def has_no_edges(self) -> bool:
        return self.edge_count == 0

    def vertices(self) -> Iterator[Vertex]:
        return iter(self._vertices.values())

    def edges(self) -> Iterator[Edge]:
        return iter(self._edges)

    def to_dict(self) -> Dict:
        return {
            'directed': self.directed,
            'vertices': list(self._vertices.keys()),
            'edges': [
                {
                    'source': e.source.value,
                    'target': e.target.value,
                    'weight': e.weight,
                    'directed': e.directed
                } for e in self._edges
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Graph':
        graph = cls(directed=data['directed'])
        for vertex in data['vertices']:
            graph.add_vertex(vertex)
        for edge in data['edges']:
            graph.add_edge(edge['source'], edge['target'], edge['weight'], edge['directed'])
        return graph

    def serialize(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, json_str: str) -> 'Graph':
        return cls.from_dict(json.loads(json_str))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vertices.clear()
        self._edges.clear()
