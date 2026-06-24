#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_domain_validation import candidate_kind, domain_pack, node_target


QUERY_PREFIXES = (
    "Find the memory nodes for",
    "Route this operational issue:",
    "Which graph memories should be used for",
    "Attach the new case about",
    "Recover the relevant policy and runbook for",
)

REGIONS = ("us-east", "us-west", "eu-central", "apac", "global")
SEVERITIES = ("low", "medium", "high", "urgent")
ACTORS = ("support lead", "operations manager", "risk analyst", "incident commander", "workflow owner")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diverse business-domain teacher episodes for Qwen labeling.")
    parser.add_argument("--output-dir", default="data/domain_teacher_episodes")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=73037)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    episodes = build_episodes(
        episode_count=args.episodes,
        candidate_limit=args.candidate_limit,
        rng=rng,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "episodes_000.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": 1,
        "kind": "domain_teacher_episodes",
        "seed": args.seed,
        "episodes": len(episodes),
        "candidate_limit": args.candidate_limit,
        "episode_files": [path.name],
        "domains": sorted({episode["domain"] for episode in episodes}),
        "description": (
            "Diverse synthetic business-domain graph episodes with hard same-domain, "
            "cross-domain, bridge, compliance, tenant, and workflow negatives. Labels "
            "are intended to be filled by Qwen teacher labeling."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(episodes)} domain teacher episodes to {output_dir}")


def build_episodes(*, episode_count: int, candidate_limit: int, rng: random.Random) -> list[dict]:
    domains = domain_pack()
    episodes = []
    for index in range(episode_count):
        domain = domains[index % len(domains)]
        query = domain["queries"][(index // len(domains)) % len(domain["queries"])]
        variant = index // (len(domains) * len(domain["queries"]))
        episodes.append(
            build_episode(
                domain=domain,
                all_domains=domains,
                query=query,
                variant=variant,
                candidate_limit=candidate_limit,
                rng=rng,
            )
        )
    return episodes


def build_episode(
    *,
    domain: dict,
    all_domains: Sequence[dict],
    query: dict,
    variant: int,
    candidate_limit: int,
    rng: random.Random,
) -> dict:
    node_by_id = {node["id"]: node for node in domain["nodes"]}
    region = REGIONS[variant % len(REGIONS)]
    severity = SEVERITIES[(variant // len(REGIONS)) % len(SEVERITIES)]
    actor = ACTORS[(variant // (len(REGIONS) * len(SEVERITIES))) % len(ACTORS)]
    query_text = query_variant_text(query["text"], region=region, severity=severity, actor=actor, rng=rng)
    expected_topic = f"{domain['id']}:{query['workflow']}"
    seed_id = query.get("bridge_ids", [domain["nodes"][0]["id"]])[0]
    seed_node = node_by_id[seed_id]
    candidate_rows = candidate_rows_for(all_domains, domain, query, candidate_limit=candidate_limit, rng=rng)

    candidates = []
    for candidate_domain, candidate in candidate_rows:
        candidate_id = candidate["id"]
        is_same_domain = candidate_domain["id"] == domain["id"]
        is_positive = is_same_domain and candidate_id in set(query["positive_ids"])
        is_bridge = is_same_domain and candidate_id in set(query.get("bridge_ids", []))
        kind = domain_candidate_kind(candidate, is_same_domain=is_same_domain, is_positive=is_positive, is_bridge=is_bridge)
        target = domain_node_target(candidate, query, is_same_domain=is_same_domain, is_positive=is_positive, is_bridge=is_bridge)
        if is_bridge:
            target = {
                "follow": 0.84,
                "read_full": 0.24,
                "include": 0.18,
                "expand": 0.94,
                "stop": 0.04,
                "result": 0.14,
            }
        candidates.append(
            {
                "id": f"{domain['id']}:v{variant:04d}:{query['id']}:{len(candidates):03d}",
                "parent_id": seed_id,
                "dst_id": global_node_id(candidate_domain, candidate),
                "kind": kind,
                "edge_summary": edge_summary(seed_node, candidate, query=query, kind=kind),
                "confidence": candidate_confidence(kind, rng),
                "hop": 0 if is_bridge else rng.choice([1, 1, 1, 2]),
                "source_topic": expected_topic,
                "destination_topic": destination_topic(candidate_domain, candidate),
                "destination_plain_topic": candidate["metadata"].get("workflow", candidate["metadata"].get("kind", "")),
                "destination_terms": destination_terms(candidate),
                "relation": relation(domain, candidate_domain, query, candidate, kind=kind, is_positive=is_positive),
                "retrieval_reason": retrieval_reason(kind, rng),
                "node_summary": candidate["summary"],
                "node_full": candidate["full"],
                "node_topic": destination_topic(candidate_domain, candidate),
                "bootstrap_teacher": {
                    key: round(float(value), 6)
                    for key, value in target.items()
                    if key in {"follow", "read_full", "include", "expand", "stop"}
                },
            }
        )

    return {
        "schema_version": 1,
        "kind": "domain_teacher_episode",
        "id": f"{domain['id']}-v{variant:04d}-{query['id']}",
        "domain": domain["id"],
        "query": query_text,
        "expected_topic": expected_topic,
        "query_intent": {
            "text": query_text,
            "target_topic": expected_topic,
            "target_plain_topic": query["workflow"],
            "target_terms": sorted(set(query["text"].lower().replace(";", " ").replace(",", " ").split())),
            "domain": domain["id"],
            "tenant": domain["tenant"],
            "workflow": query["workflow"],
            "region": region,
            "severity": severity,
            "actor": actor,
            "metadata": query.get("metadata", {}),
            "expected_positive_ids": query["positive_ids"],
            "hard_negative_ids": query.get("hard_negative_ids", []),
            "negative_policy": (
                "Reject candidates that only share domain words but differ by workflow, tenant, "
                "jurisdiction, entity type, customer scope, safety level, or incident event."
            ),
        },
        "seed_ids": [seed_id],
        "path": [
            {
                "node_id": seed_id,
                "summary": seed_node["summary"],
                "topic": destination_topic(domain, seed_node),
                "plain_topic": seed_node["metadata"].get("workflow", seed_node["metadata"].get("kind", "")),
                "terms": destination_terms(seed_node),
            }
        ],
        "current_node": {
            "node_id": seed_id,
            "summary": seed_node["summary"],
            "full": seed_node["full"],
            "topic": destination_topic(domain, seed_node),
            "plain_topic": seed_node["metadata"].get("workflow", seed_node["metadata"].get("kind", "")),
            "terms": destination_terms(seed_node),
        },
        "candidates": candidates,
        "teacher_prompt": {
            "instruction": (
                "Label business-domain memory graph candidates. Use metadata scope and full text, "
                "not lexical overlap alone. Same-domain wrong workflow candidates are hard negatives."
            )
        },
    }


def query_variant_text(base: str, *, region: str, severity: str, actor: str, rng: random.Random) -> str:
    prefix = rng.choice(QUERY_PREFIXES)
    suffix = rng.choice(
        [
            f"Region={region}; severity={severity}; owner={actor}.",
            f"The {actor} needs the right memory path for {region} with {severity} urgency.",
            f"Use tenant/workflow scope carefully; this is {severity} in {region}.",
        ]
    )
    return f"{prefix} {base} {suffix}"


def domain_candidate_kind(candidate: dict, *, is_same_domain: bool, is_positive: bool, is_bridge: bool) -> str:
    if not is_same_domain:
        return "cross_domain_negative"
    return candidate_kind(candidate, is_positive=is_positive, is_bridge=is_bridge)


def domain_node_target(
    candidate: dict,
    query: dict,
    *,
    is_same_domain: bool,
    is_positive: bool,
    is_bridge: bool,
) -> dict[str, float]:
    if not is_same_domain:
        return {
            "follow": 0.04,
            "read_full": 0.12,
            "include": 0.02,
            "expand": 0.04,
            "stop": 0.90,
            "result": 0.01,
        }
    return node_target(candidate, query, is_positive=is_positive)


def candidate_rows_for(
    all_domains: Sequence[dict],
    domain: dict,
    query: dict,
    *,
    candidate_limit: int,
    rng: random.Random,
) -> list[tuple[dict, dict]]:
    node_by_id = {node["id"]: node for node in domain["nodes"]}
    selected: list[tuple[dict, dict]] = []
    selected_ids: set[str] = set()

    def add_many(ids: Sequence[str]) -> None:
        for node_id in ids:
            key = global_node_id(domain, node_by_id[node_id])
            if key not in selected_ids:
                selected.append((domain, node_by_id[node_id]))
                selected_ids.add(key)

    def add_external(nodes: Sequence[tuple[dict, dict]]) -> None:
        for candidate_domain, node in nodes:
            key = global_node_id(candidate_domain, node)
            if key not in selected_ids:
                selected.append((candidate_domain, node))
                selected_ids.add(key)

    positives = list(query["positive_ids"])
    hard = list(query.get("hard_negative_ids", []))
    bridge = list(query.get("bridge_ids", []))
    rng.shuffle(positives)
    rng.shuffle(hard)
    add_many(positives)
    add_many(bridge)
    add_many(hard)

    same_domain = [
        node["id"]
        for node in domain["nodes"]
        if global_node_id(domain, node) not in selected_ids and node["kind"] != "cross_domain_negative"
    ]
    cross_domain = [
        (other_domain, node)
        for other_domain in all_domains
        for node in other_domain["nodes"]
        if other_domain["id"] != domain["id"]
    ]
    rng.shuffle(same_domain)
    rng.shuffle(cross_domain)
    add_many(same_domain[: max(3, candidate_limit // 3)])
    add_external(cross_domain[: max(6, candidate_limit // 2)])
    add_many([node["id"] for node in domain["nodes"]])
    add_external(cross_domain)
    return selected[:candidate_limit]


def global_node_id(domain: dict, candidate: dict) -> str:
    return f"{domain['id']}:{candidate['id']}"


def destination_topic(domain: dict, candidate: dict) -> str:
    metadata = candidate["metadata"]
    workflow = metadata.get("workflow", metadata.get("kind", "unknown"))
    return f"{domain['id']}:{workflow}"


def destination_terms(candidate: dict) -> list[str]:
    metadata = candidate["metadata"]
    values = [
        str(metadata.get("kind", "")),
        str(metadata.get("workflow", "")),
        str(metadata.get("entity", "")),
        str(metadata.get("account_tier", "")),
        str(metadata.get("risk_signal", "")),
        str(metadata.get("care_flow", "")),
        str(metadata.get("event", "")),
        str(metadata.get("service", "")),
    ]
    values.extend(candidate["summary"].lower().replace(";", " ").replace(",", " ").split()[:8])
    return sorted({value for value in values if value})


def edge_summary(seed_node: dict, candidate: dict, *, query: dict, kind: str) -> str:
    return (
        f"{seed_node['metadata'].get('workflow', seed_node['metadata'].get('kind', 'seed'))} -> "
        f"{candidate['metadata'].get('workflow', candidate['metadata'].get('kind', 'candidate'))}; "
        f"query_workflow={query['workflow']}; candidate_kind={kind}"
    )


def relation(
    source_domain: dict,
    candidate_domain: dict,
    query: dict,
    candidate: dict,
    *,
    kind: str,
    is_positive: bool,
) -> dict[str, object]:
    candidate_workflow = candidate["metadata"].get("workflow", candidate["metadata"].get("kind", ""))
    return {
        "domain": candidate_domain["id"],
        "source_domain": source_domain["id"],
        "tenant": candidate_domain["tenant"],
        "source_tenant": source_domain["tenant"],
        "query_workflow": query["workflow"],
        "candidate_workflow": candidate_workflow,
        "same_workflow": candidate_workflow == query["workflow"],
        "same_domain": candidate_domain["id"] == source_domain["id"],
        "same_tenant": candidate_domain["tenant"] == source_domain["tenant"],
        "is_expected_positive": is_positive,
        "candidate_kind": kind,
        "metadata_conflict": metadata_conflict(query, candidate),
        "decision_hint": decision_hint(kind),
    }


def metadata_conflict(query: dict, candidate: dict) -> list[str]:
    conflicts = []
    query_metadata = query.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    for key in ("account_tier", "entity", "risk_signal", "care_flow", "event", "service"):
        if key in query_metadata and key in candidate_metadata and query_metadata[key] != candidate_metadata[key]:
            conflicts.append(key)
    return conflicts


def decision_hint(kind: str) -> str:
    if kind == "positive":
        return "target_answer"
    if kind == "bridge_positive":
        return "traversal_bridge_not_answer"
    if "hard" in kind:
        return "same_domain_wrong_scope_hard_negative"
    return "off_domain_or_unrelated_negative"


def retrieval_reason(kind: str, rng: random.Random) -> str:
    if kind == "positive":
        return rng.choice(["target_workflow_candidate", "metadata_scope_match", "full_text_policy_match"])
    if kind == "bridge_positive":
        return "expressway_or_hub_candidate"
    if "hard" in kind:
        return rng.choice(["lexical_overlap_wrong_scope", "same_domain_hard_negative", "metadata_conflict_candidate"])
    return rng.choice(["cross_domain_distractor", "semantic_near_miss", "random_index_candidate"])


def candidate_confidence(kind: str, rng: random.Random) -> float:
    if kind == "positive":
        return round(rng.uniform(0.78, 0.96), 6)
    if kind == "bridge_positive":
        return round(rng.uniform(0.84, 0.98), 6)
    if "hard" in kind:
        return round(rng.uniform(0.50, 0.82), 6)
    return round(rng.uniform(0.20, 0.62), 6)


if __name__ == "__main__":
    main()
