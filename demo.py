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


def make_node(node_id: str, summary: str, full: str) -> NodeFrame:
    return NodeFrame(
        node_id=node_id,
        summary_vector=embed_text(summary, 64),
        full_vector=embed_text(full, 128),
        summary_payload=summary,
        full_payload=full,
    )


def connect(store: GraphStore, src_id: str, dst_id: str, confidence: float = 0.9) -> None:
    src = store.get_node(src_id)
    dst = store.get_node(dst_id)
    store.add_edge(
        EdgeFrame(
            src_id=src_id,
            dst_id=dst_id,
            edge_vector=stable_edge_vector(src.summary_vector, dst.summary_vector, 32),
            confidence=confidence,
        )
    )


def main() -> None:
    store = GraphStore(max_outgoing_edges=16)
    scorer = HeuristicTraversalScorer()

    nodes = [
        make_node("python", "Python programming language", "Python is used for scripting, ML, APIs, and tooling."),
        make_node("torch", "PyTorch tensor neural network library", "PyTorch trains neural networks with tensors."),
        make_node("bert", "BERT transformer encoder model", "BERT is a transformer encoder for scoring text or vector frames."),
        make_node("graph", "Vector graph traversal system", "The graph stores node vectors and edge vectors."),
        make_node("sqlite", "SQLite embedded database", "SQLite stores local structured data in one file."),
        make_node("ann", "Approximate nearest neighbor vector search", "ANN finds seed nodes from query vectors."),
    ]
    for node in nodes:
        store.add_node(node)

    connect(store, "python", "torch")
    connect(store, "torch", "bert")
    connect(store, "bert", "graph")
    connect(store, "graph", "ann")
    connect(store, "python", "sqlite", confidence=0.65)
    connect(store, "ann", "graph")

    query = embed_text("small transformer traverses vector graph edges", 64)
    controller = TraversalController(
        store=store,
        scorer=scorer,
        config=TraversalConfig(max_hops=3, fanout=8, beam_width=8, max_visited=64),
    )
    result = controller.traverse(query_vector=query, seed_id="python")

    print("Query traversal from seed 'python'")
    print("Included:")
    for decision in sorted(result.included, key=lambda item: (-item.include_score, item.node_id)):
        print(
            f"  {decision.node_id:8s} "
            f"include={decision.include_score:.3f} "
            f"follow={decision.follow_score:.3f} "
            f"read_full={decision.read_full} "
            f"path={' -> '.join(decision.path)}"
        )

    new_node = make_node(
        "edge_frames",
        "Compact edge vector frames for learned relationships",
        "Edge frames hold relationship geometry as a compact vector without symbolic relation types.",
    )
    attached = insert_node(
        store=store,
        scorer=scorer,
        node=new_node,
        seed_id="graph",
        config=InsertConfig(attach_limit=4, attach_threshold=0.55),
    )
    print()
    print(f"Inserted 'edge_frames' and attached to: {', '.join(attached) or '(none)'}")


if __name__ == "__main__":
    main()

