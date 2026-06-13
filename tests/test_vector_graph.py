import unittest

from vector_graph import (
    EdgeFrame,
    GraphStore,
    HeuristicTraversalScorer,
    InsertConfig,
    NodeFrame,
    TraversalConfig,
    TraversalController,
    embed_text,
    insert_node,
)
from vector_graph.vectors import stable_edge_vector


def node(node_id: str, text: str) -> NodeFrame:
    return NodeFrame(
        node_id=node_id,
        summary_vector=embed_text(text, 32),
        full_vector=embed_text(text + " details", 64),
        summary_payload=text,
    )


def edge(store: GraphStore, src_id: str, dst_id: str, confidence: float = 0.9) -> None:
    src = store.get_node(src_id)
    dst = store.get_node(dst_id)
    store.add_edge(
        EdgeFrame(
            src_id=src_id,
            dst_id=dst_id,
            edge_vector=stable_edge_vector(src.summary_vector, dst.summary_vector, 16),
            confidence=confidence,
        )
    )


class VectorGraphTests(unittest.TestCase):
    def make_store(self) -> GraphStore:
        store = GraphStore(max_outgoing_edges=4)
        for frame in [
            node("a", "vector graph root"),
            node("b", "transformer traversal scorer"),
            node("c", "edge frame relationship vector"),
            node("d", "unrelated storage database"),
        ]:
            store.add_node(frame)
        edge(store, "a", "b", 0.95)
        edge(store, "b", "c", 0.95)
        edge(store, "a", "d", 0.50)
        return store

    def test_traversal_is_deterministic(self) -> None:
        store = self.make_store()
        scorer = HeuristicTraversalScorer()
        controller = TraversalController(
            store=store,
            scorer=scorer,
            config=TraversalConfig(max_hops=3, fanout=4, beam_width=4),
        )
        query = embed_text("transformer follows edge vector frames", 32)

        first = controller.traverse(query_vector=query, seed_id="a")
        second = controller.traverse(query_vector=query, seed_id="a")

        self.assertEqual(first, second)

    def test_edge_cap_is_enforced(self) -> None:
        store = GraphStore(max_outgoing_edges=2)
        store.add_node(node("root", "root"))
        for index in range(4):
            child = node(f"child_{index}", f"child {index}")
            store.add_node(child)
            edge(store, "root", child.node_id, confidence=0.4 + index / 10)

        self.assertEqual(len(store.get_edges("root")), 2)
        self.assertEqual([item.dst_id for item in store.get_edges("root")], ["child_3", "child_2"])

    def test_insert_node_creates_bounded_attachments(self) -> None:
        store = self.make_store()
        scorer = HeuristicTraversalScorer()
        inserted = node("new", "compact edge vector relationship frame")

        attached = insert_node(
            store=store,
            scorer=scorer,
            node=inserted,
            seed_id="a",
            config=InsertConfig(attach_limit=2, attach_threshold=0.0),
        )

        self.assertLessEqual(len(attached), 2)
        self.assertGreaterEqual(len(store.get_edges("new")), 1)

    def test_max_hops_is_respected(self) -> None:
        store = self.make_store()
        scorer = HeuristicTraversalScorer()
        controller = TraversalController(
            store=store,
            scorer=scorer,
            config=TraversalConfig(max_hops=1, fanout=4, beam_width=4, expand_threshold=0.0),
        )
        query = embed_text("edge frame relationship vector", 32)

        result = controller.traverse(query_vector=query, seed_id="a")

        self.assertTrue(result.visited)
        self.assertTrue(all(decision.hop <= 1 for decision in result.visited))


if __name__ == "__main__":
    unittest.main()
