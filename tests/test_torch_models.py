import unittest

from vector_graph import EdgeFrame, GraphStore, NodeFrame, TraversalConfig, TraversalController, embed_text
from vector_graph.vectors import stable_edge_vector


class TorchModelTests(unittest.TestCase):
    def test_torch_scorer_runs_deterministically(self) -> None:
        try:
            from vector_graph.torch_models import TorchModelConfig, TorchTraversalScorer
        except ImportError as exc:
            self.skipTest(str(exc))

        store = GraphStore(max_outgoing_edges=4)
        root = NodeFrame(
            node_id="root",
            summary_vector=embed_text("root vector graph", 32),
            full_vector=embed_text("root vector graph full", 64),
        )
        child = NodeFrame(
            node_id="child",
            summary_vector=embed_text("child traversal model", 32),
            full_vector=embed_text("child traversal model full", 64),
        )
        store.add_node(root)
        store.add_node(child)
        store.add_edge(
            EdgeFrame(
                src_id="root",
                dst_id="child",
                edge_vector=stable_edge_vector(root.summary_vector, child.summary_vector, 16),
                confidence=0.9,
            )
        )

        config = TorchModelConfig(query_dim=32, summary_dim=32, edge_dim=16, full_dim=64, path_dim=32)
        first_scorer = TorchTraversalScorer.initialized(config, seed=7)
        second_scorer = TorchTraversalScorer.initialized(config, seed=7)
        controller_config = TraversalConfig(max_hops=1, include_threshold=0.0, expand_threshold=1.0)
        query = embed_text("traversal model", 32)

        first = TraversalController(
            store=store,
            scorer=first_scorer,
            config=controller_config,
        ).traverse(query_vector=query, seed_id="root")
        second = TraversalController(
            store=store,
            scorer=second_scorer,
            config=controller_config,
        ).traverse(query_vector=query, seed_id="root")

        self.assertEqual(first, second)
        self.assertEqual(len(first.visited), 1)


if __name__ == "__main__":
    unittest.main()

