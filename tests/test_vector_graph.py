import unittest
from collections import Counter
from typing import Sequence

from vector_graph import (
    EdgeFrame,
    GraphStore,
    HeuristicTraversalScorer,
    InsertConfig,
    NodeFrame,
    TraversalConfig,
    TraversalController,
    TraversalIndex,
    TraversalIndexConfig,
    embed_text,
    insert_node,
)
from vector_graph.frames import TraversalScores
from vector_graph.vectors import stable_edge_vector


def node(node_id: str, text: str, *, expressway: bool = False) -> NodeFrame:
    return NodeFrame(
        node_id=node_id,
        summary_vector=embed_text(text, 32),
        full_vector=embed_text(text + " details", 64),
        summary_payload=text,
        metadata={"expressway": True} if expressway else {},
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


class FixedTraversalScorer:
    def __init__(self, scores: dict[str, TraversalScores]) -> None:
        self.scores = scores

    def score_edge(
        self,
        *,
        query_vector: Sequence[float],
        current_node: NodeFrame,
        edge: EdgeFrame,
        dst_node: NodeFrame,
        path_vector: Sequence[float],
        hop: int,
    ) -> TraversalScores:
        return self.scores.get(
            edge.dst_id,
            TraversalScores(
                follow_score=0.1,
                read_full_score=0.1,
                include_score=0.1,
                expand_score=0.0,
                stop_score=1.0,
            ),
        )

    def score_attach(
        self,
        *,
        new_node: NodeFrame,
        candidate_node: NodeFrame,
        path_vector: Sequence[float],
    ) -> float:
        return 0.0


class BatchAttachScorer(FixedTraversalScorer):
    def __init__(self, scores: dict[str, TraversalScores]) -> None:
        super().__init__(scores)
        self.attach_batches = 0
        self.attach_calls = 0

    def score_attach(
        self,
        *,
        new_node: NodeFrame,
        candidate_node: NodeFrame,
        path_vector: Sequence[float],
    ) -> float:
        self.attach_calls += 1
        return 0.9

    def score_attach_batch(
        self,
        *,
        new_node: NodeFrame,
        candidate_nodes: Sequence[NodeFrame],
        path_vectors: Sequence[Sequence[float]],
    ) -> tuple[float, ...]:
        self.attach_batches += 1
        return tuple(0.9 for _ in candidate_nodes)


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

    def test_expressway_edge_cap_is_larger_than_normal_cap(self) -> None:
        store = GraphStore(max_outgoing_edges=2, max_expressway_edges=5)
        store.add_node(node("hub", "expressway hub", expressway=True))
        store.add_node(node("root", "normal root"))

        for index in range(6):
            child = node(f"hub_child_{index}", f"hub child {index}")
            store.add_node(child)
            edge(store, "hub", child.node_id, confidence=0.4 + index / 20)

        for index in range(4):
            child = node(f"root_child_{index}", f"root child {index}")
            store.add_node(child)
            edge(store, "root", child.node_id, confidence=0.4 + index / 20)

        self.assertEqual(len(store.get_edges("hub")), 5)
        self.assertEqual(len(store.get_edges("root")), 2)

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

    def test_insert_node_uses_batch_attach_scoring_when_available(self) -> None:
        store = self.make_store()
        scorer = BatchAttachScorer({})
        inserted = node("new", "compact edge vector relationship frame")

        attached = insert_node(
            store=store,
            scorer=scorer,
            node=inserted,
            seed_id="a",
            config=InsertConfig(attach_limit=3, attach_threshold=0.5),
        )

        self.assertTrue(attached)
        self.assertEqual(scorer.attach_batches, 1)
        self.assertEqual(scorer.attach_calls, 0)

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

    def test_single_path_commits_one_candidate_per_hop(self) -> None:
        store = GraphStore(max_outgoing_edges=4)
        for frame in [
            node("root", "root"),
            node("strong", "best next step"),
            node("weak", "plausible but weaker"),
            node("leaf", "final useful node"),
        ]:
            store.add_node(frame)
        edge(store, "root", "weak", 0.99)
        edge(store, "root", "strong", 0.90)
        edge(store, "strong", "leaf", 0.90)
        scorer = FixedTraversalScorer(
            {
                "strong": TraversalScores(0.9, 0.4, 0.7, 0.8, 0.1),
                "weak": TraversalScores(0.7, 0.4, 0.7, 0.8, 0.1),
                "leaf": TraversalScores(0.8, 0.4, 0.7, 0.0, 0.1),
            }
        )
        controller = TraversalController(
            store=store,
            scorer=scorer,
            config=TraversalConfig(
                max_hops=3,
                fanout=4,
                beam_width=4,
                mode="single_path",
                include_threshold=0.0,
                expand_threshold=0.5,
            ),
        )

        result = controller.traverse(query_vector=embed_text("best next step", 32), seed_id="root")

        self.assertEqual({decision.node_id for decision in result.visited}, {"strong", "leaf"})
        self.assertNotIn("weak", {decision.node_id for decision in result.visited})
        self.assertTrue(all(count == 1 for count in Counter(decision.hop for decision in result.visited).values()))

    def test_critical_score_can_force_inclusion(self) -> None:
        store = GraphStore(max_outgoing_edges=2)
        store.add_node(node("root", "root"))
        store.add_node(node("critical", "low summary high full detail"))
        edge(store, "root", "critical", 0.90)
        scorer = FixedTraversalScorer(
            {
                "critical": TraversalScores(
                    follow_score=0.2,
                    read_full_score=0.9,
                    include_score=0.1,
                    expand_score=0.0,
                    stop_score=0.9,
                )
            }
        )
        controller = TraversalController(
            store=store,
            scorer=scorer,
            config=TraversalConfig(max_hops=1, include_threshold=1.0, critical_threshold=0.8),
        )

        result = controller.traverse(query_vector=embed_text("detail", 32), seed_id="root")

        decision = result.visited[0]
        self.assertTrue(decision.critical)
        self.assertTrue(decision.included)
        self.assertEqual(result.rejected, ())

    def test_included_results_are_sorted_by_result_score(self) -> None:
        store = GraphStore(max_outgoing_edges=4)
        for frame in [
            node("root", "root"),
            node("bridge", "high follow bridge"),
            node("answer", "best returned answer"),
            node("support", "supporting answer"),
        ]:
            store.add_node(frame)
        edge(store, "root", "bridge", 0.99)
        edge(store, "root", "answer", 0.90)
        edge(store, "root", "support", 0.95)
        scorer = FixedTraversalScorer(
            {
                "bridge": TraversalScores(0.95, 0.2, 0.7, 0.0, 0.2, result_score=0.2),
                "answer": TraversalScores(0.7, 0.9, 0.8, 0.0, 0.2, result_score=0.95),
                "support": TraversalScores(0.8, 0.7, 0.8, 0.0, 0.2, result_score=0.75),
            }
        )
        controller = TraversalController(
            store=store,
            scorer=scorer,
            config=TraversalConfig(max_hops=1, fanout=4, beam_width=4, include_threshold=0.0),
        )

        result = controller.traverse(query_vector=embed_text("best returned answer", 32), seed_id="root")

        self.assertEqual([decision.node_id for decision in result.included], ["answer", "support", "bridge"])

    def test_expressway_node_can_expand_above_normal_threshold(self) -> None:
        store = GraphStore(max_outgoing_edges=2, max_expressway_edges=4)
        store.add_node(node("root", "root"))
        store.add_node(node("hub", "expressway hub", expressway=True))
        store.add_node(node("target", "far target"))
        edge(store, "root", "hub", 0.90)
        edge(store, "hub", "target", 0.90)
        scorer = FixedTraversalScorer(
            {
                "hub": TraversalScores(0.9, 0.2, 0.2, 0.0, 0.2),
                "target": TraversalScores(0.8, 0.2, 0.7, 0.0, 0.2),
            }
        )
        controller = TraversalController(
            store=store,
            scorer=scorer,
            config=TraversalConfig(
                max_hops=2,
                fanout=2,
                beam_width=2,
                expand_threshold=1.0,
                expressway_threshold=0.8,
                max_expressway_jumps=1,
            ),
        )

        result = controller.traverse(query_vector=embed_text("far target", 32), seed_id="root")

        hub = next(decision for decision in result.visited if decision.node_id == "hub")
        self.assertTrue(hub.expressway)
        self.assertEqual(hub.expressway_jumps, 1)
        self.assertIn("target", {decision.node_id for decision in result.visited})

    def test_traversal_index_is_deterministic_and_bounded(self) -> None:
        frames = [
            node("alpha", "vector traversal target"),
            node("beta", "database storage unrelated"),
            node("gamma", "edge frame relationship"),
            node("delta", "transformer scoring model"),
        ]
        config = TraversalIndexConfig(dimension=16, table_count=6, bits_per_table=4, seed=123)
        first = TraversalIndex(config=config)
        second = TraversalIndex(config=config)
        first.add_nodes(frames)
        second.add_nodes(reversed(frames))
        query = embed_text("vector traversal target", 32)

        first_hits = first.query(query, limit=3)
        second_hits = second.query(query, limit=3)

        self.assertEqual(first_hits, second_hits)
        self.assertLessEqual(len(first_hits), 3)
        self.assertEqual(first_hits[0].node_id, "alpha")

    def test_traversal_index_uses_compact_metadata_vector(self) -> None:
        target_vector = [1.0] + [0.0] * 15
        distractor_vector = [-1.0] + [0.0] * 15
        frames = [
            NodeFrame(
                node_id="target",
                summary_vector=embed_text("unrelated summary", 32),
                metadata={"traversal_vector": target_vector},
            ),
            NodeFrame(
                node_id="distractor",
                summary_vector=embed_text("target words in summary", 32),
                metadata={"traversal_vector": distractor_vector},
            ),
        ]
        index = TraversalIndex(config=TraversalIndexConfig(dimension=16, table_count=4, bits_per_table=4, seed=7))
        index.add_nodes(frames)

        self.assertEqual(index.seed_ids(target_vector, limit=1), ("target",))

    def test_traversal_index_removes_replaced_nodes(self) -> None:
        old_vector = [1.0] + [0.0] * 15
        new_vector = [-1.0] + [0.0] * 15
        index = TraversalIndex(config=TraversalIndexConfig(dimension=16, table_count=4, bits_per_table=4, seed=7))

        index.add_node(
            NodeFrame(
                node_id="moving",
                summary_vector=embed_text("moving node", 32),
                metadata={"traversal_vector": old_vector},
            )
        )
        self.assertEqual(index.seed_ids(old_vector, limit=1), ("moving",))

        index.add_node(
            NodeFrame(
                node_id="moving",
                summary_vector=embed_text("moving node", 32),
                metadata={"traversal_vector": new_vector},
            )
        )

        self.assertEqual(index.seed_ids(new_vector, limit=1), ("moving",))
        self.assertEqual(index.seed_ids(old_vector, limit=1), ())

    def test_traversal_can_start_from_index_seed_ids(self) -> None:
        store = GraphStore(max_outgoing_edges=2)
        for frame in [
            node("root", "cold start root"),
            node("indexed", "indexed useful region"),
            node("target", "target reached from index"),
        ]:
            store.add_node(frame)
        edge(store, "indexed", "target", 0.90)
        scorer = FixedTraversalScorer(
            {
                "target": TraversalScores(
                    follow_score=0.9,
                    read_full_score=0.2,
                    include_score=0.8,
                    expand_score=0.0,
                    stop_score=0.2,
                )
            }
        )
        controller = TraversalController(
            store=store,
            scorer=scorer,
            config=TraversalConfig(max_hops=1, fanout=2, beam_width=2),
        )

        result = controller.traverse(
            query_vector=embed_text("target reached from index", 32),
            seed_id="root",
            extra_seed_ids=("indexed",),
        )

        self.assertEqual([decision.node_id for decision in result.visited], ["target"])


if __name__ == "__main__":
    unittest.main()
