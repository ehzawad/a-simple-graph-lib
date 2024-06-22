import unittest
from graph_core import Graph, Vertex, Edge, VertexNotFoundError, EdgeNotFoundError, InvalidOperationError

class TestGraphAdvanced(unittest.TestCase):

    def setUp(self):
        self.graph = Graph(directed=True)

    def test_self_loops(self):
        self.graph.add_edge("A", "A", weight=2.0)
        self.assertEqual(self.graph.vertex_count, 1)
        self.assertEqual(self.graph.edge_count, 1)
        self.assertEqual(self.graph.get_vertex("A").degree, 1)

    def test_parallel_edges(self):
        self.graph.add_edge("A", "B", weight=1.0)
        self.graph.add_edge("A", "B", weight=2.0)
        self.assertEqual(self.graph.edge_count, 1)
        self.assertEqual(self.graph.get_edge("A", "B").weight, 2.0)

    def test_undirected_graph(self):
        undirected = Graph(directed=False)
        undirected.add_edge("A", "B")
        self.assertTrue(undirected.are_adjacent("A", "B"))
        self.assertTrue(undirected.are_adjacent("B", "A"))

    def test_mixed_graph(self):
        self.graph.add_edge("A", "B", directed=True)
        self.graph.add_edge("B", "C", directed=False)
        self.assertTrue(self.graph.are_adjacent("A", "B"))
        self.assertFalse(self.graph.are_adjacent("B", "A"))
        self.assertTrue(self.graph.are_adjacent("B", "C"))
        self.assertTrue(self.graph.are_adjacent("C", "B"))

    def test_vertex_properties(self):
        self.graph.add_edge("A", "B")
        self.graph.add_edge("A", "C")
        self.graph.add_edge("D", "A")
        vertex_a = self.graph.get_vertex("A")
        self.assertEqual(vertex_a.in_degree, 1)
        self.assertEqual(vertex_a.out_degree, 2)
        self.assertFalse(vertex_a.is_leaf)
        self.assertFalse(vertex_a.is_root)

        vertex_d = self.graph.get_vertex("D")
        self.assertTrue(vertex_d.is_root)

        vertex_c = self.graph.get_vertex("C")
        self.assertTrue(vertex_c.is_leaf)

    def test_edge_properties(self):
        edge = self.graph.add_edge("A", "B", weight=3.14, directed=True)
        self.assertEqual(edge.weight, 3.14)
        self.assertTrue(edge.directed)

    def test_graph_properties(self):
        self.assertTrue(self.graph.is_empty())
        self.assertTrue(self.graph.has_no_edges())
        self.graph.add_vertex("A")
        self.assertFalse(self.graph.is_empty())
        self.assertTrue(self.graph.has_no_edges())
        self.graph.add_edge("A", "B")
        self.assertFalse(self.graph.has_no_edges())

    def test_graph_operations(self):
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        self.graph.add_edge("C", "A")
        self.assertTrue(self.graph.has_cycle())
        self.assertFalse(self.graph.is_tree())

        tree = Graph(directed=True)
        tree.add_edge("A", "B")
        tree.add_edge("A", "C")
        tree.add_edge("B", "D")
        tree.add_edge("B", "E")
        self.assertFalse(tree.has_cycle())
        self.assertTrue(tree.is_tree())

    def test_serialization(self):
        self.graph.add_edge("A", "B", weight=1.5)
        self.graph.add_edge("B", "C", weight=2.0, directed=False)
        serialized = self.graph.serialize()
        deserialized = Graph.deserialize(serialized)
        self.assertEqual(self.graph.vertex_count, deserialized.vertex_count)
        self.assertEqual(self.graph.edge_count, deserialized.edge_count)
        self.assertEqual(self.graph.directed, deserialized.directed)
        self.assertEqual(self.graph.get_edge("A", "B").weight, deserialized.get_edge("A", "B").weight)
        self.assertEqual(self.graph.get_edge("B", "C").directed, deserialized.get_edge("B", "C").directed)

    def test_graph_exceptions(self):
        with self.assertRaises(VertexNotFoundError):
            self.graph.remove_vertex("A")

        with self.assertRaises(EdgeNotFoundError):
            self.graph.remove_edge("A", "B")

        # Add more exception tests as needed

    def test_strongly_connected_components_edge_cases(self):
        # Test with an empty graph
        self.assertEqual(self.graph.strongly_connected_components(), [])

        # Test with a single vertex
        self.graph.add_vertex("A")
        self.assertEqual(self.graph.strongly_connected_components(), [["A"]])

        # Test with two disconnected vertices
        self.graph.add_vertex("B")
        self.assertEqual(sorted(map(sorted, self.graph.strongly_connected_components())), [["A"], ["B"]])

    def test_graph_context_manager(self):
        with Graph(directed=True) as g:
            g.add_edge("A", "B")
            g.add_edge("B", "C")
            self.assertEqual(g.vertex_count, 3)
            self.assertEqual(g.edge_count, 2)

        # After exiting the context, the graph should be cleared
        self.assertEqual(g.vertex_count, 0)
        self.assertEqual(g.edge_count, 0)

if __name__ == '__main__':
    unittest.main()
