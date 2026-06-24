import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def test_batch_scores_match_single_scores_and_checkpoint_loads(self) -> None:
        try:
            import torch
            from vector_graph.torch_models import TorchModelConfig, TorchTraversalScorer
        except ImportError as exc:
            self.skipTest(str(exc))

        store = GraphStore(max_outgoing_edges=8)
        root = NodeFrame(node_id="root", summary_vector=embed_text("root graph", 32))
        children = [
            NodeFrame(node_id="a", summary_vector=embed_text("alpha traversal", 32)),
            NodeFrame(node_id="b", summary_vector=embed_text("beta edge", 32)),
            NodeFrame(node_id="c", summary_vector=embed_text("gamma node", 32)),
        ]
        store.add_node(root)
        for child in children:
            store.add_node(child)
            store.add_edge(
                EdgeFrame(
                    src_id="root",
                    dst_id=child.node_id,
                    edge_vector=stable_edge_vector(root.summary_vector, child.summary_vector, 16),
                    confidence=0.8,
                )
            )

        config = TorchModelConfig(query_dim=32, summary_dim=32, edge_dim=16, full_dim=64, path_dim=32)
        scorer = TorchTraversalScorer.initialized(config, seed=11)
        query = embed_text("alpha edge traversal", 32)
        edges = store.get_edges("root")
        dst_nodes = [store.get_node(edge.dst_id) for edge in edges]
        batch = scorer.score_edges(
            query_vector=query,
            current_node=root,
            edges=edges,
            dst_nodes=dst_nodes,
            path_vector=root.summary_vector,
            hop=0,
        )
        singles = tuple(
            scorer.score_edge(
                query_vector=query,
                current_node=root,
                edge=edge,
                dst_node=dst,
                path_vector=root.summary_vector,
                hop=0,
            )
            for edge, dst in zip(edges, dst_nodes)
        )
        self.assertEqual(len(batch), len(singles))
        for batch_score, single_score in zip(batch, singles):
            for field in (
                "follow_score",
                "read_full_score",
                "include_score",
                "expand_score",
                "stop_score",
                "result_score",
            ):
                self.assertAlmostEqual(getattr(batch_score, field), getattr(single_score, field), places=6)

        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "scorer.pt"
            torch.save(
                {
                    "config": config.__dict__,
                    "traversal_model": scorer.traversal_model.state_dict(),
                    "attach_model": scorer.attach_model.state_dict(),
                },
                checkpoint,
            )
            loaded = TorchTraversalScorer.from_checkpoint(checkpoint)
            self.assertEqual(
                loaded.score_edges(
                    query_vector=query,
                    current_node=root,
                    edges=edges,
                    dst_nodes=dst_nodes,
                    path_vector=root.summary_vector,
                    hop=0,
                ),
                batch,
            )

    def test_transformer_scorer_runs_and_checkpoint_loads(self) -> None:
        try:
            import torch
            from vector_graph.torch_models import TorchModelConfig, TorchTraversalScorer
        except ImportError as exc:
            self.skipTest(str(exc))

        store = GraphStore(max_outgoing_edges=4)
        root = NodeFrame(node_id="root", summary_vector=embed_text("root graph", 32))
        child = NodeFrame(node_id="child", summary_vector=embed_text("transformer scorer", 32))
        store.add_node(root)
        store.add_node(child)
        store.add_edge(
            EdgeFrame(
                src_id="root",
                dst_id="child",
                edge_vector=stable_edge_vector(root.summary_vector, child.summary_vector, 16),
                confidence=0.75,
            )
        )

        config = TorchModelConfig(
            query_dim=32,
            summary_dim=32,
            edge_dim=16,
            full_dim=64,
            path_dim=32,
            model_kind="transformer",
        )
        scorer = TorchTraversalScorer.initialized(config, seed=17)
        query = embed_text("transformer traversal", 32)
        edge = store.get_edges("root")[0]
        child_node = store.get_node("child")
        score = scorer.score_edge(
            query_vector=query,
            current_node=root,
            edge=edge,
            dst_node=child_node,
            path_vector=root.summary_vector,
            hop=0,
        )
        self.assertGreaterEqual(score.follow_score, 0.0)
        self.assertLessEqual(score.follow_score, 1.0)

        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "transformer.pt"
            torch.save(
                {
                    "config": config.__dict__,
                    "traversal_model": scorer.traversal_model.state_dict(),
                    "attach_model": scorer.attach_model.state_dict(),
                },
                checkpoint,
            )
            loaded = TorchTraversalScorer.from_checkpoint(checkpoint)
            loaded_score = loaded.score_edge(
                query_vector=query,
                current_node=root,
                edge=edge,
                dst_node=child_node,
                path_vector=root.summary_vector,
                hop=0,
            )
            self.assertEqual(score, loaded_score)

    def test_hybrid_attach_head_scores(self) -> None:
        try:
            from vector_graph.torch_models import TorchModelConfig, TorchTraversalScorer
        except ImportError as exc:
            self.skipTest(str(exc))

        config = TorchModelConfig(
            query_dim=32,
            summary_dim=32,
            edge_dim=16,
            full_dim=64,
            path_dim=32,
            model_kind="transformer",
            attach_head_kind="hybrid",
        )
        scorer = TorchTraversalScorer.initialized(config, seed=23)
        new_node = NodeFrame(
            node_id="new",
            summary_vector=embed_text("new attach node", 32),
            full_vector=embed_text("new attach node full detail", 64),
        )
        candidate_node = NodeFrame(
            node_id="candidate",
            summary_vector=embed_text("candidate attach node", 32),
            full_vector=embed_text("candidate attach node full detail", 64),
        )
        score = scorer.score_attach(
            new_node=new_node,
            candidate_node=candidate_node,
            path_vector=embed_text("attach path", 32),
        )
        batch_score = scorer.score_attach_batch(
            new_node=new_node,
            candidate_nodes=[candidate_node],
            path_vectors=[embed_text("attach path", 32)],
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(batch_score, (score,))

    def test_listwise_loss_prefers_positive_rank_mass(self) -> None:
        try:
            import torch
            from scripts.train_scorer import listwise_softmax_loss
        except ImportError as exc:
            self.skipTest(str(exc))

        labels = torch.tensor([[1.0, 0.0, 0.0]])
        good_scores = torch.tensor([[3.0, 1.0, 0.0]])
        bad_scores = torch.tensor([[0.0, 1.0, 3.0]])

        self.assertLess(
            float(listwise_softmax_loss(good_scores, labels)),
            float(listwise_softmax_loss(bad_scores, labels)),
        )

    def test_pairwise_loss_uses_continuous_teacher_order(self) -> None:
        try:
            import torch
            from scripts.train_scorer import pairwise_margin_loss
        except ImportError as exc:
            self.skipTest(str(exc))

        labels = torch.tensor([[0.9, 0.4, 0.1]])
        good_scores = torch.tensor([[0.8, 0.5, 0.2]])
        bad_scores = torch.tensor([[0.2, 0.5, 0.8]])

        self.assertLess(
            float(pairwise_margin_loss(good_scores, labels, margin=0.15, min_delta=0.1)),
            float(pairwise_margin_loss(bad_scores, labels, margin=0.15, min_delta=0.1)),
        )

    def test_teacher_converter_separates_follow_and_include_rank_targets(self) -> None:
        from scripts.convert_teacher_episodes import write_ranking_files

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            episode = {
                "id": "episode-1",
                "query": "find include only node",
                "expected_topic": "topic:a",
                "current_node": {"summary": "current"},
                "path": [{"summary": "current"}],
                "candidates": [
                    {
                        "id": "candidate-1",
                        "kind": "positive",
                        "node_summary": "include only",
                        "node_full": "include only full",
                        "confidence": 0.9,
                        "hop": 0,
                        "qwen_teacher": {
                            "follow": 0.0,
                            "read_full": 0.8,
                            "include": 0.9,
                            "expand": 0.7,
                            "stop": 0.0,
                        },
                    }
                ],
            }

            write_ranking_files([episode], ranking_dir=output_dir, teacher_key="qwen_teacher")
            traversal = next(read_jsonl(output_dir / "traversal_ranking.jsonl"))
            attach = next(read_jsonl(output_dir / "attach_ranking.jsonl"))

        self.assertEqual(traversal["candidates"][0]["label"], 0)
        self.assertEqual(traversal["candidates"][0]["rank_target"], 0.0)
        self.assertEqual(attach["candidates"][0]["label"], 1)
        self.assertEqual(attach["candidates"][0]["rank_target"], 0.9)

    def test_qwen_labeler_shard_helpers_are_deterministic(self) -> None:
        from scripts.label_teacher_episodes_qwen import default_output_name, episode_in_shard

        self.assertEqual(default_output_name(None), "episodes_000.jsonl")
        self.assertEqual(default_output_name(7), "episodes_007.jsonl")
        self.assertEqual(
            [index for index in range(10) if episode_in_shard(index, shard_index=1, shard_count=3)],
            [1, 4, 7],
        )

    def test_qwen_prompt_includes_relationship_metadata(self) -> None:
        from scripts.label_teacher_episodes_qwen import build_prompt

        prompt = build_prompt(
            {
                "id": "episode-1",
                "query": "find training label issue",
                "expected_topic": "domain:training",
                "query_intent": {"target_topic": "domain:training"},
                "path": [
                    {
                        "node_id": "n1",
                        "summary": "training path",
                        "topic": "domain:training",
                        "plain_topic": "training",
                        "terms": ["label"],
                    }
                ],
                "current_node": {
                    "node_id": "n1",
                    "summary": "training path",
                    "full": "full training path",
                    "topic": "domain:training",
                    "plain_topic": "training",
                    "terms": ["label"],
                },
                "candidates": [
                    {
                        "id": "candidate-1",
                        "parent_id": "n1",
                        "dst_id": "n2",
                        "kind": "ambiguous_wrong_topic",
                        "source_topic": "domain:training",
                        "destination_topic": "domain:security",
                        "destination_plain_topic": "security",
                        "destination_terms": ["policy", "audit"],
                        "edge_summary": "training -> security",
                        "confidence": 0.4,
                        "hop": 0,
                        "relation": {
                            "same_as_query_topic": False,
                            "decision_hint": "lexically_ambiguous_wrong_topic",
                        },
                        "node_summary": "security label wording",
                        "node_full": "security label wording full",
                    }
                ],
            }
        )

        self.assertIn('"expected_topic":"domain:training"', prompt)
        self.assertIn('"destination_topic":"domain:security"', prompt)
        self.assertIn('"decision_hint":"lexically_ambiguous_wrong_topic"', prompt)
        self.assertIn("hard negative", prompt)

    def test_qwen_labeler_marks_quota_errors_non_retryable(self) -> None:
        from scripts.label_teacher_episodes_qwen import is_non_retryable_http_error

        self.assertTrue(
            is_non_retryable_http_error(
                403,
                '{"error":{"code":"AllocationQuota.FreeTierOnly","message":"free tier exhausted"}}',
            )
        )
        self.assertTrue(is_non_retryable_http_error(401, '{"error":{"message":"bad key"}}'))
        self.assertFalse(is_non_retryable_http_error(429, '{"error":{"message":"rate limited"}}'))
        self.assertFalse(is_non_retryable_http_error(500, '{"error":{"message":"server failed"}}'))

    def test_qwen_label_parser_ignores_hallucinated_extra_candidate_ids(self) -> None:
        from scripts.label_teacher_episodes_qwen import parse_labels

        labels = parse_labels(
            json.dumps(
                {
                    "candidates": [
                        {
                            "id": "candidate-0",
                            "follow": 0.2,
                            "read_full": 0.8,
                            "include": 0.7,
                            "expand": 0.3,
                            "stop": 0.1,
                        },
                        {
                            "id": "candidate-1",
                            "follow": 0.1,
                            "read_full": 0.4,
                            "include": 0.2,
                            "expand": 0.0,
                            "stop": 0.9,
                        },
                        {
                            "id": "candidate-2",
                            "follow": 1.0,
                            "read_full": 1.0,
                            "include": 1.0,
                            "expand": 1.0,
                            "stop": 0.0,
                        },
                    ]
                }
            ),
            expected_ids=["candidate-0", "candidate-1"],
        )

        self.assertEqual(set(labels), {"candidate-0", "candidate-1"})
        self.assertEqual(labels["candidate-0"]["include"], 0.7)

    def test_qwen_shard_runner_marks_short_success_as_partial(self) -> None:
        from scripts.run_qwen_label_shards import classify_shard_status

        self.assertEqual(
            classify_shard_status(returncode=0, line_count=8, expected_per_shard=8),
            "complete",
        )
        self.assertEqual(
            classify_shard_status(returncode=0, line_count=7, expected_per_shard=8),
            "partial",
        )
        self.assertEqual(
            classify_shard_status(returncode=1, line_count=8, expected_per_shard=8),
            "error",
        )

    def test_benchmark_metrics_track_hard_negatives_and_calibration(self) -> None:
        from scripts.benchmark_scorer import ranking_metrics

        metrics = ranking_metrics(
            cases=[{"id": "case-1"}],
            spans=[(0, 3)],
            predictions=[0.9, 0.1, 0.2],
            labels=[1, 0, 0],
            kinds=["positive", "easy_negative", "lexical_hard_negative_candidate"],
            oracle_scores=[0.95, 0.05, 0.1],
        )

        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["hard_negative_pairwise_accuracy"], 1.0)
        self.assertLess(metrics["brier_score"], 0.1)
        self.assertIn("lexical_hard_negative_candidate", metrics["pairwise_accuracy_by_negative_kind"])

    def test_benchmark_target_pairwise_sampling_is_bounded_and_deterministic(self) -> None:
        from scripts.benchmark_scorer import target_pairwise_metrics

        scores = [index / 1000 for index in range(1000)]
        targets = [index / 1000 for index in range(1000)]

        first = target_pairwise_metrics(scores, targets, max_pairs=100)
        second = target_pairwise_metrics(scores, targets, max_pairs=100)

        self.assertEqual(first, second)
        self.assertEqual(first["sampled"], 1)
        self.assertLessEqual(first["pairs"], 100)
        self.assertEqual(first["accuracy"], 1.0)

    def test_benchmark_traversal_action_metrics_use_teacher_oracle_vector(self) -> None:
        try:
            import torch
            from torch import nn
            from scripts.benchmark_scorer import evaluate_traversal
        except ImportError as exc:
            self.skipTest(str(exc))

        class FixedTraversalModel(nn.Module):
            def forward(self, rows):  # type: ignore[no-untyped-def]
                return torch.tensor(
                    [
                        [0.9, 0.1, 0.8, 0.7, 0.1],
                        [0.1, 0.9, 0.2, 0.2, 0.8],
                    ],
                    dtype=torch.float32,
                )

        zero32 = [0.0] * 32
        zero16 = [0.0] * 16
        cases = [
            {
                "id": "case-1",
                "query": zero32,
                "current_summary": zero32,
                "path": zero32,
                "candidates": [
                    {
                        "id": "good",
                        "kind": "positive",
                        "label": 1,
                        "rank_target": 0.9,
                        "oracle": [0.9, 0.1, 0.8, 0.7, 0.1],
                        "dst_summary": zero32,
                        "edge": zero16,
                        "confidence": 0.9,
                        "hop": 0,
                    },
                    {
                        "id": "bad",
                        "kind": "hard_negative",
                        "label": 0,
                        "rank_target": 0.1,
                        "oracle": [0.1, 0.9, 0.2, 0.2, 0.8],
                        "dst_summary": zero32,
                        "edge": zero16,
                        "confidence": 0.4,
                        "hop": 1,
                    },
                ],
            }
        ]

        metrics = evaluate_traversal(
            FixedTraversalModel(),
            cases,
            device=torch.device("cpu"),
            batch_size=2,
            positive_threshold=0.65,
        )

        self.assertEqual(metrics["precision_at_1"], 1.0)
        self.assertEqual(metrics["action_metrics"]["follow"]["average_precision"], 1.0)
        self.assertEqual(metrics["action_metrics"]["include"]["average_precision"], 1.0)
        self.assertEqual(metrics["action_metrics"]["stop"]["average_precision"], 1.0)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                import json

                yield json.loads(stripped)


if __name__ == "__main__":
    unittest.main()
