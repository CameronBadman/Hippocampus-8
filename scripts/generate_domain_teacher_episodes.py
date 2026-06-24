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
    parser.add_argument(
        "--domain-set",
        choices=("curated", "broad", "all"),
        default="curated",
        help="Use the original curated domain pack, the broader generated pack, or both.",
    )
    parser.add_argument("--seed", type=int, default=73037)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    domains = domain_pack_for(args.domain_set)
    episodes = build_episodes(
        episode_count=args.episodes,
        candidate_limit=args.candidate_limit,
        rng=rng,
        domains=domains,
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
        "domain_set": args.domain_set,
        "episode_files": [path.name],
        "domains": sorted({episode["domain"] for episode in episodes}),
        "domain_count": len({episode["domain"] for episode in episodes}),
        "workflow_count": len({episode["expected_topic"] for episode in episodes}),
        "description": (
            "Diverse synthetic business-domain graph episodes with hard same-domain, "
            "cross-domain, bridge, compliance, tenant, and workflow negatives. Labels "
            "are intended to be filled by Qwen teacher labeling."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(episodes)} domain teacher episodes to {output_dir}")


def build_episodes(
    *,
    episode_count: int,
    candidate_limit: int,
    rng: random.Random,
    domains: Sequence[dict] | None = None,
) -> list[dict]:
    if domains is None:
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


def domain_pack_for(domain_set: str) -> list[dict]:
    curated = domain_pack()
    if domain_set == "curated":
        return curated
    broad = broad_domain_pack()
    if domain_set == "broad":
        return broad
    if domain_set == "all":
        return curated + broad
    raise ValueError(f"unknown domain set: {domain_set}")


def broad_domain_pack() -> list[dict]:
    return [_build_broad_domain(spec) for spec in BROAD_DOMAIN_SPECS]


def _build_broad_domain(spec: dict) -> dict:
    hub_id = f"{spec['id']}_hub"
    workflows = spec["workflows"]
    nodes = [
        _node(
            hub_id,
            spec["hub_summary"],
            spec["hub_full"],
            {"kind": "hub", "workflow": spec["hub_workflow"]},
            "bridge_positive",
            confidence=0.93,
            hop=0,
            requires_full_read=False,
        )
    ]
    for workflow in workflows:
        for suffix, label, full_hint in (
            ("policy", "policy", "decision thresholds, exclusions, ownership, and approval rules"),
            ("runbook", "runbook", "step-by-step triage, escalation, communications, and closure checks"),
            ("evidence", "evidence", "required records, audit fields, retention, and verification artifacts"),
        ):
            node_id = f"{workflow['id']}_{suffix}"
            nodes.append(
                _node(
                    node_id,
                    f"{workflow['title']} {label}",
                    (
                        f"{workflow['title']} {label} for {spec['label']}. "
                        f"Use when {workflow['use_when']}. Covers {full_hint}. "
                        f"Do not use for {workflow['not_for']}."
                    ),
                    {
                        "kind": label,
                        "workflow": workflow["workflow"],
                        **workflow["metadata"],
                    },
                    "positive",
                )
            )

    for decoy in spec.get("decoys", ()):
        nodes.append(
            _node(
                decoy["id"],
                decoy["summary"],
                decoy["full"],
                {"kind": decoy.get("kind", "reference"), "workflow": decoy["workflow"], **decoy.get("metadata", {})},
                decoy.get("node_kind", "hard_same_domain_negative"),
            )
        )

    queries = []
    workflow_ids = [workflow["id"] for workflow in workflows]
    for index, workflow in enumerate(workflows):
        next_one = workflows[(index + 1) % len(workflows)]
        next_two = workflows[(index + 2) % len(workflows)]
        hard_negative_ids = [
            f"{next_one['id']}_policy",
            f"{next_one['id']}_runbook",
            f"{next_two['id']}_policy",
        ]
        if spec.get("decoys"):
            hard_negative_ids.append(spec["decoys"][index % len(spec["decoys"])]["id"])
        queries.append(
            {
                "id": workflow["id"],
                "workflow": workflow["workflow"],
                "text": workflow["query"],
                "expected_use": workflow["expected_use"],
                "metadata": workflow["metadata"],
                "positive_ids": [
                    f"{workflow['id']}_policy",
                    f"{workflow['id']}_runbook",
                    f"{workflow['id']}_evidence",
                ],
                "bridge_ids": [hub_id],
                "hard_negative_ids": hard_negative_ids,
                "path_hint": " ".join([workflow["workflow"], *workflow_ids[max(0, index - 1) : index + 2]]),
            }
        )

    return {
        "id": spec["id"],
        "tenant": spec["tenant"],
        "seed_summary": spec["seed_summary"],
        "queries": queries,
        "nodes": nodes,
    }


def _node(
    node_id: str,
    summary: str,
    full: str,
    metadata: dict,
    kind: str,
    *,
    confidence: float = 0.84,
    hop: int = 1,
    requires_full_read: bool = True,
) -> dict:
    return {
        "id": node_id,
        "summary": summary,
        "full": full,
        "metadata": metadata,
        "kind": kind,
        "confidence": confidence,
        "hop": hop,
        "requires_full_read": requires_full_read,
    }


BROAD_DOMAIN_SPECS = (
    {
        "id": "legal_contract_ops",
        "tenant": "sterling_legal",
        "label": "legal contract operations",
        "seed_summary": "Legal operations memory for contracts, privacy addenda, renewal terms, employment clauses, and dispute notices.",
        "hub_workflow": "contract_ops",
        "hub_summary": "Contract operations routing hub",
        "hub_full": "Routes commercial legal requests by clause family, jurisdiction, counterparty type, renewal stage, and evidence requirements.",
        "workflows": [
            {
                "id": "vendor_dpa_review",
                "workflow": "privacy_addendum_review",
                "title": "Vendor data processing addendum review",
                "query": "A vendor DPA lists new subprocessors and EU transfer terms; find privacy addendum review guidance.",
                "expected_use": "attach vendor privacy request to DPA review, subprocessor evidence, and transfer-risk policy",
                "metadata": {"jurisdiction": "eu", "entity": "vendor", "risk_signal": "subprocessor_change"},
                "use_when": "a vendor or processor changes privacy terms, subprocessors, or international transfer language",
                "not_for": "sales discount approvals, employment offer clauses, or routine renewal billing notices",
            },
            {
                "id": "renewal_price_cap",
                "workflow": "commercial_renewal",
                "title": "Commercial renewal price-cap clause review",
                "query": "An enterprise renewal has an auto-renewal price cap conflict; find commercial clause and approval guidance.",
                "expected_use": "attach renewal negotiation to price-cap clause review and approval evidence",
                "metadata": {"account_tier": "enterprise", "entity": "customer", "risk_signal": "price_cap"},
                "use_when": "renewal language changes price caps, auto-renewal terms, or commercial approvals",
                "not_for": "privacy subprocessors, employment offers, or litigation hold notices",
            },
            {
                "id": "litigation_hold_notice",
                "workflow": "litigation_hold",
                "title": "Litigation hold notice and preservation workflow",
                "query": "A customer dispute may require preserving emails and support logs; find litigation hold steps.",
                "expected_use": "attach dispute notice to preservation policy, evidence checklist, and legal owner workflow",
                "metadata": {"event": "customer_dispute", "entity": "customer", "risk_signal": "preservation"},
                "use_when": "a dispute, subpoena, or threatened claim requires preserving records",
                "not_for": "commercial price cap negotiation or routine privacy addendum review",
            },
            {
                "id": "employment_clause_exception",
                "workflow": "employment_contract_exception",
                "title": "Employment contract exception review",
                "query": "A senior hire requests non-standard severance and invention assignment terms; find employment contract guidance.",
                "expected_use": "attach employment exception to approval, clause library, and HR/legal evidence",
                "metadata": {"entity": "employee", "risk_signal": "non_standard_clause"},
                "use_when": "offer letters or employment agreements include non-standard legal terms",
                "not_for": "customer litigation preservation or vendor privacy terms",
            },
        ],
        "decoys": [
            {
                "id": "nda_counterparty_address",
                "summary": "NDA counterparty address update",
                "full": "Administrative update for NDA party address fields; does not change privacy, renewal, litigation, or employment obligations.",
                "workflow": "nda_admin",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "cybersecurity_response",
        "tenant": "nightwatch_security",
        "label": "cybersecurity incident response",
        "seed_summary": "Security operations memory for phishing, endpoint compromise, vulnerability response, cloud identity, and disclosure workflows.",
        "hub_workflow": "security_response",
        "hub_summary": "Security response routing hub",
        "hub_full": "Routes security investigations by event type, affected asset, severity, evidence, and disclosure obligations.",
        "workflows": [
            {
                "id": "phishing_oauth_consent",
                "workflow": "phishing_identity_response",
                "title": "Phishing OAuth consent response",
                "query": "A user approved a suspicious OAuth app after a phishing email; find containment and evidence steps.",
                "expected_use": "attach identity phishing case to OAuth revocation, mailbox review, and user notification workflow",
                "metadata": {"event": "phishing", "service": "identity", "risk_signal": "oauth_consent"},
                "use_when": "phishing produces mailbox compromise, OAuth grants, or suspicious sign-in sessions",
                "not_for": "container CVEs, endpoint malware, or vendor questionnaire evidence",
            },
            {
                "id": "critical_cve_exposure",
                "workflow": "vulnerability_response",
                "title": "Critical CVE internet exposure response",
                "query": "An internet-facing service is affected by a critical CVE; find exposure validation and patch escalation.",
                "expected_use": "attach vulnerability alert to exposure triage, compensating controls, and patch SLA",
                "metadata": {"event": "cve", "service": "edge", "severity": "critical"},
                "use_when": "a vulnerability affects externally reachable infrastructure or exploitable package versions",
                "not_for": "phishing OAuth grants or endpoint malware isolation",
            },
            {
                "id": "endpoint_malware_isolation",
                "workflow": "endpoint_containment",
                "title": "Endpoint malware isolation workflow",
                "query": "An employee laptop shows malware beaconing and suspicious persistence; find isolation and forensics guidance.",
                "expected_use": "attach endpoint alert to network isolation, evidence capture, and reimage approval",
                "metadata": {"event": "malware", "service": "endpoint", "risk_signal": "persistence"},
                "use_when": "endpoint telemetry shows malware execution, persistence, or command-and-control traffic",
                "not_for": "cloud IAM policy drift or public CVE patch planning",
            },
            {
                "id": "cloud_iam_privilege_drift",
                "workflow": "cloud_identity_review",
                "title": "Cloud IAM privilege drift review",
                "query": "A production role gained broad admin privileges outside the change window; find review and rollback steps.",
                "expected_use": "attach IAM drift to privilege review, approval evidence, and rollback control",
                "metadata": {"event": "privilege_drift", "service": "cloud_iam", "risk_signal": "admin_role"},
                "use_when": "cloud roles, policies, or service accounts gain unexpected privileges",
                "not_for": "endpoint malware cleanup or phishing mailbox triage",
            },
        ],
        "decoys": [
            {
                "id": "security_awareness_calendar",
                "summary": "Security awareness training calendar",
                "full": "Training schedule for awareness campaigns; not an incident response, vulnerability, endpoint, or IAM playbook.",
                "workflow": "training",
                "kind": "training",
            }
        ],
    },
    {
        "id": "insurance_claims_ops",
        "tenant": "harbor_mutual",
        "label": "insurance claims operations",
        "seed_summary": "Insurance claims memory for property damage, liability review, fraud flags, subrogation, and catastrophe handling.",
        "hub_workflow": "claims_ops",
        "hub_summary": "Claims operations routing hub",
        "hub_full": "Routes claims by line of business, loss event, coverage question, fraud signal, and evidence completeness.",
        "workflows": [
            {
                "id": "water_damage_coverage",
                "workflow": "property_coverage_review",
                "title": "Property water damage coverage review",
                "query": "A homeowner reports water damage after a burst pipe; find coverage review and adjuster evidence.",
                "expected_use": "attach property claim to coverage checks, damage evidence, and adjuster workflow",
                "metadata": {"entity": "policyholder", "event": "water_damage", "line": "property"},
                "use_when": "property claims involve water damage, exclusions, mitigation, and adjuster documentation",
                "not_for": "auto liability disputes, medical fraud flags, or subrogation demand letters",
            },
            {
                "id": "auto_liability_dispute",
                "workflow": "auto_liability_review",
                "title": "Auto liability dispute review",
                "query": "Two drivers dispute fault after a rear-end accident with conflicting statements; find liability review guidance.",
                "expected_use": "attach auto claim to fault review, statement evidence, and settlement authority",
                "metadata": {"line": "auto", "event": "collision", "risk_signal": "fault_dispute"},
                "use_when": "auto claims require fault allocation, witness statements, and settlement review",
                "not_for": "home water damage exclusions or healthcare claim fraud scoring",
            },
            {
                "id": "medical_claim_fraud_flag",
                "workflow": "fraud_special_investigation",
                "title": "Medical claim fraud special investigation",
                "query": "A medical claim has duplicate provider billing and unusual treatment codes; find SIU review steps.",
                "expected_use": "attach suspicious medical claim to fraud triage, evidence retention, and SIU escalation",
                "metadata": {"line": "health", "risk_signal": "duplicate_billing", "entity": "provider"},
                "use_when": "claim patterns indicate fraud, duplicate billing, staged loss, or provider anomalies",
                "not_for": "routine auto liability or property coverage adjustment",
            },
            {
                "id": "subrogation_recovery",
                "workflow": "subrogation_recovery",
                "title": "Subrogation recovery workflow",
                "query": "A third-party contractor caused a covered loss; find subrogation notice and recovery evidence.",
                "expected_use": "attach claim to subrogation demand, recovery evidence, and legal handoff workflow",
                "metadata": {"event": "third_party_fault", "entity": "contractor", "line": "property"},
                "use_when": "another party may owe recovery after the carrier pays a covered loss",
                "not_for": "first-party coverage review or SIU fraud intake",
            },
        ],
        "decoys": [
            {
                "id": "policy_address_change",
                "summary": "Policy mailing address change",
                "full": "Administrative customer address update unrelated to claim coverage, liability, fraud, or subrogation handling.",
                "workflow": "policy_admin",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "hr_people_ops",
        "tenant": "northstar_people",
        "label": "HR and people operations",
        "seed_summary": "People operations memory for onboarding, leave accommodations, performance review, payroll corrections, and employee relations.",
        "hub_workflow": "people_ops",
        "hub_summary": "People operations routing hub",
        "hub_full": "Routes HR cases by employee lifecycle stage, policy sensitivity, payroll impact, and documentation requirements.",
        "workflows": [
            {
                "id": "leave_accommodation",
                "workflow": "leave_accommodation",
                "title": "Leave accommodation review",
                "query": "An employee requested medical leave accommodation with intermittent schedule changes; find HR review guidance.",
                "expected_use": "attach accommodation case to leave policy, documentation, and manager communication workflow",
                "metadata": {"entity": "employee", "risk_signal": "medical_leave", "jurisdiction": "us"},
                "use_when": "employees request leave, accommodation, schedule adjustment, or medical documentation review",
                "not_for": "payroll corrections, onboarding equipment, or performance calibration",
            },
            {
                "id": "payroll_correction",
                "workflow": "payroll_correction",
                "title": "Payroll correction workflow",
                "query": "A bonus was missed in payroll close and needs correction before financial reporting; find payroll steps.",
                "expected_use": "attach compensation issue to payroll correction, approval evidence, and employee notice",
                "metadata": {"event": "missed_bonus", "entity": "employee", "service": "payroll"},
                "use_when": "pay, bonus, deduction, or tax withholding errors need correction",
                "not_for": "medical accommodation review or onboarding access provisioning",
            },
            {
                "id": "onboarding_access",
                "workflow": "new_hire_onboarding",
                "title": "New hire onboarding access workflow",
                "query": "A new hire starts next week and needs laptop, account provisioning, and compliance tasks; find onboarding guidance.",
                "expected_use": "attach onboarding case to access checklist, equipment fulfillment, and policy acknowledgements",
                "metadata": {"event": "new_hire", "entity": "employee", "service": "access"},
                "use_when": "new employees need accounts, equipment, training, or policy acknowledgements",
                "not_for": "payroll adjustments or employee relations investigations",
            },
            {
                "id": "employee_relations_investigation",
                "workflow": "employee_relations",
                "title": "Employee relations investigation workflow",
                "query": "A manager reported harassment allegations that need confidential intake and investigation steps.",
                "expected_use": "attach ER case to intake protocol, confidentiality rules, and investigation evidence",
                "metadata": {"event": "harassment_report", "entity": "employee", "risk_signal": "confidential"},
                "use_when": "workplace conduct allegations require confidential HR investigation",
                "not_for": "routine performance review calibration or onboarding checklists",
            },
        ],
        "decoys": [
            {
                "id": "office_snack_budget",
                "summary": "Office snack budget approval",
                "full": "Facilities budget note for office snacks; unrelated to leave, payroll, onboarding, or employee relations cases.",
                "workflow": "facilities_admin",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "retail_store_ops",
        "tenant": "marketlane_retail",
        "label": "retail store operations",
        "seed_summary": "Retail operations memory for store incidents, inventory shrink, promotions, returns, and point-of-sale exceptions.",
        "hub_workflow": "store_ops",
        "hub_summary": "Retail store operations routing hub",
        "hub_full": "Routes store operations cases by customer impact, inventory movement, promotion rules, loss prevention, and POS state.",
        "workflows": [
            {
                "id": "pos_outage_queue",
                "workflow": "pos_outage",
                "title": "Point-of-sale outage queue workflow",
                "query": "A store cannot process card payments during peak hours; find POS outage and customer queue guidance.",
                "expected_use": "attach store outage to POS fallback, escalation, and customer communication procedure",
                "metadata": {"service": "pos", "event": "outage", "severity": "high"},
                "use_when": "checkout systems fail, payment terminals are down, or store queues need fallback handling",
                "not_for": "promotion pricing disputes or inventory shrink investigations",
            },
            {
                "id": "inventory_shrink_review",
                "workflow": "loss_prevention",
                "title": "Inventory shrink loss-prevention review",
                "query": "Cycle counts show repeated high-value item shrink at one store; find loss-prevention review steps.",
                "expected_use": "attach shrink case to evidence review, camera request, and inventory adjustment policy",
                "metadata": {"event": "shrink", "inventory_class": "high_value", "risk_signal": "loss_prevention"},
                "use_when": "inventory variance suggests theft, process loss, or repeated shrink patterns",
                "not_for": "POS outages or customer return exceptions",
            },
            {
                "id": "promotion_price_override",
                "workflow": "promotion_pricing",
                "title": "Promotion price override workflow",
                "query": "A regional promotion is scanning at the wrong price; find override and margin approval guidance.",
                "expected_use": "attach promotion issue to price override, regional approval, and customer remediation",
                "metadata": {"event": "price_mismatch", "service": "promotion", "region": "regional"},
                "use_when": "promotion rules, price books, or regional discounts mismatch expected checkout price",
                "not_for": "loss prevention or warranty return triage",
            },
            {
                "id": "warranty_return_exception",
                "workflow": "return_exception",
                "title": "Warranty return exception workflow",
                "query": "A customer return is outside the window but may qualify for warranty exception; find return guidance.",
                "expected_use": "attach return exception to warranty policy, manager approval, and restock handling",
                "metadata": {"event": "return_exception", "entity": "customer", "risk_signal": "warranty"},
                "use_when": "returns exceed normal policy but warranty, defect, or customer exception may apply",
                "not_for": "promotion price override or POS payment outage",
            },
        ],
        "decoys": [
            {
                "id": "store_playlist_update",
                "summary": "Store playlist update",
                "full": "Marketing note for in-store music rotation; unrelated to POS, inventory, promotion, or return workflows.",
                "workflow": "marketing_admin",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "education_admin_ops",
        "tenant": "lakeside_university",
        "label": "education administration",
        "seed_summary": "Education administration memory for admissions, financial aid, academic integrity, accessibility, and registrar exceptions.",
        "hub_workflow": "student_admin",
        "hub_summary": "Student administration routing hub",
        "hub_full": "Routes student administration cases by academic term, student status, policy family, privacy requirements, and evidence.",
        "workflows": [
            {
                "id": "financial_aid_verification",
                "workflow": "financial_aid_verification",
                "title": "Financial aid verification workflow",
                "query": "A student's aid package is held for income verification near payment deadline; find verification steps.",
                "expected_use": "attach aid case to document checklist, deadline exception, and student notice workflow",
                "metadata": {"entity": "student", "event": "aid_hold", "term": "fall"},
                "use_when": "financial aid requires income, dependency, or eligibility verification",
                "not_for": "academic misconduct review or disability accommodation intake",
            },
            {
                "id": "academic_integrity_case",
                "workflow": "academic_integrity",
                "title": "Academic integrity case review",
                "query": "A professor reported suspected exam collaboration; find academic integrity review and evidence steps.",
                "expected_use": "attach misconduct report to evidence intake, student notice, and hearing procedure",
                "metadata": {"event": "misconduct_report", "entity": "student", "risk_signal": "exam_collaboration"},
                "use_when": "academic misconduct, plagiarism, or exam integrity concerns require formal review",
                "not_for": "financial aid income verification or registrar transcript requests",
            },
            {
                "id": "accessibility_accommodation",
                "workflow": "student_accommodation",
                "title": "Student accessibility accommodation workflow",
                "query": "A student requests testing accommodations after submitting documentation; find accessibility review guidance.",
                "expected_use": "attach accommodation case to documentation review, faculty notice, and privacy handling",
                "metadata": {"entity": "student", "risk_signal": "accessibility", "service": "testing"},
                "use_when": "student accommodations, accessible materials, or testing adjustments are requested",
                "not_for": "academic integrity cases or admissions waitlist decisions",
            },
            {
                "id": "registrar_late_drop",
                "workflow": "registrar_exception",
                "title": "Registrar late drop exception workflow",
                "query": "A student requests a late course drop after medical disruption; find registrar exception policy.",
                "expected_use": "attach late drop case to petition review, transcript impact, and documentation checklist",
                "metadata": {"event": "late_drop", "entity": "student", "term": "current"},
                "use_when": "course drops, withdrawals, transcript exceptions, or enrollment status changes need review",
                "not_for": "financial aid verification or accessibility testing accommodation",
            },
        ],
        "decoys": [
            {
                "id": "campus_parking_notice",
                "summary": "Campus parking notice",
                "full": "General campus parking communication unrelated to aid, integrity, accommodation, or registrar exceptions.",
                "workflow": "campus_admin",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "energy_utility_ops",
        "tenant": "gridline_energy",
        "label": "energy utility operations",
        "seed_summary": "Utility operations memory for outage restoration, meter disputes, vegetation risk, interconnection, and safety incidents.",
        "hub_workflow": "utility_ops",
        "hub_summary": "Utility operations routing hub",
        "hub_full": "Routes utility cases by customer impact, grid asset, safety risk, regulatory obligation, and field evidence.",
        "workflows": [
            {
                "id": "storm_outage_restoration",
                "workflow": "outage_restoration",
                "title": "Storm outage restoration workflow",
                "query": "A storm caused feeder outages with critical customers affected; find restoration prioritization guidance.",
                "expected_use": "attach outage event to restoration priority, crew dispatch, and customer notification procedure",
                "metadata": {"event": "storm_outage", "service": "distribution", "severity": "critical"},
                "use_when": "weather or equipment failures produce customer outages requiring restoration sequencing",
                "not_for": "billing meter disputes or solar interconnection review",
            },
            {
                "id": "meter_billing_dispute",
                "workflow": "meter_dispute",
                "title": "Meter billing dispute review",
                "query": "A customer disputes a high bill after smart meter replacement; find meter investigation steps.",
                "expected_use": "attach billing dispute to meter read validation, field test, and customer response policy",
                "metadata": {"event": "billing_dispute", "service": "meter", "entity": "customer"},
                "use_when": "billing disputes require meter read validation, field testing, or adjustment review",
                "not_for": "storm restoration or vegetation hazard dispatch",
            },
            {
                "id": "vegetation_hazard",
                "workflow": "vegetation_risk",
                "title": "Vegetation hazard dispatch workflow",
                "query": "A tree limb is near a primary line after a customer report; find vegetation risk and dispatch guidance.",
                "expected_use": "attach hazard report to safety triage, field dispatch, and clearance evidence",
                "metadata": {"event": "vegetation_hazard", "service": "distribution", "risk_signal": "line_clearance"},
                "use_when": "trees, limbs, or vegetation threaten power lines or field crew safety",
                "not_for": "meter billing disputes or solar interconnection approvals",
            },
            {
                "id": "solar_interconnection",
                "workflow": "distributed_generation",
                "title": "Solar interconnection approval workflow",
                "query": "A rooftop solar application needs transformer capacity review before interconnection approval.",
                "expected_use": "attach interconnection case to capacity study, application evidence, and approval policy",
                "metadata": {"event": "solar_application", "service": "interconnection", "entity": "customer"},
                "use_when": "distributed generation applications require engineering and approval review",
                "not_for": "outage restoration or emergency vegetation dispatch",
            },
        ],
        "decoys": [
            {
                "id": "newsletter_efficiency_tip",
                "summary": "Customer newsletter efficiency tip",
                "full": "Marketing content for energy-saving tips; unrelated to outage, meter, vegetation, or interconnection operations.",
                "workflow": "marketing",
                "kind": "marketing",
            }
        ],
    },
    {
        "id": "hospitality_travel_ops",
        "tenant": "aurora_travel",
        "label": "hospitality and travel operations",
        "seed_summary": "Travel operations memory for hotel overbooking, loyalty exceptions, flight disruption, refunds, and guest safety.",
        "hub_workflow": "travel_ops",
        "hub_summary": "Travel operations routing hub",
        "hub_full": "Routes travel cases by supplier, guest impact, loyalty status, disruption type, refund exposure, and safety concern.",
        "workflows": [
            {
                "id": "hotel_overbooking_reaccommodation",
                "workflow": "hotel_reaccommodation",
                "title": "Hotel overbooking reaccommodation workflow",
                "query": "A loyalty guest arrived to an overbooked hotel and needs relocation compensation; find reaccommodation policy.",
                "expected_use": "attach overbooking case to relocation, compensation, and supplier escalation workflow",
                "metadata": {"event": "overbooking", "entity": "guest", "account_tier": "loyalty"},
                "use_when": "hotel inventory mismatch or overbooking requires guest relocation and compensation",
                "not_for": "flight weather waiver or chargeback dispute evidence",
            },
            {
                "id": "flight_disruption_waiver",
                "workflow": "flight_disruption",
                "title": "Flight disruption waiver workflow",
                "query": "A weather event caused flight cancellation and customers need rebooking waiver guidance.",
                "expected_use": "attach disruption case to waiver policy, rebooking steps, and customer messaging",
                "metadata": {"event": "weather_cancel", "service": "flight", "severity": "high"},
                "use_when": "flights are cancelled or delayed by weather, operational events, or carrier disruption",
                "not_for": "hotel overbooking relocation or loyalty points exception",
            },
            {
                "id": "loyalty_points_exception",
                "workflow": "loyalty_exception",
                "title": "Loyalty points exception workflow",
                "query": "A loyalty member is missing points from a partner stay; find exception and evidence requirements.",
                "expected_use": "attach loyalty case to points adjustment, partner evidence, and fraud checks",
                "metadata": {"event": "missing_points", "entity": "guest", "account_tier": "loyalty"},
                "use_when": "loyalty points, status credits, or partner accrual exceptions need adjustment",
                "not_for": "flight cancellation waivers or guest safety incidents",
            },
            {
                "id": "guest_safety_incident",
                "workflow": "guest_safety",
                "title": "Guest safety incident workflow",
                "query": "A guest reported injury at a partner property; find safety incident documentation and escalation steps.",
                "expected_use": "attach safety report to incident intake, evidence retention, and legal escalation",
                "metadata": {"event": "guest_injury", "entity": "guest", "risk_signal": "safety"},
                "use_when": "guest injury, property safety, or serious complaint requires escalation",
                "not_for": "routine points adjustment or flight waiver processing",
            },
        ],
        "decoys": [
            {
                "id": "destination_blog_update",
                "summary": "Destination blog content update",
                "full": "Editorial travel blog update unrelated to hotel, flight, loyalty, or guest safety operations.",
                "workflow": "content",
                "kind": "marketing",
            }
        ],
    },
    {
        "id": "media_ad_ops",
        "tenant": "pixelwave_media",
        "label": "media and advertising operations",
        "seed_summary": "Ad operations memory for campaign pacing, brand safety, creative approval, billing reconciliation, and measurement disputes.",
        "hub_workflow": "ad_ops",
        "hub_summary": "Advertising operations routing hub",
        "hub_full": "Routes ad operations cases by campaign, inventory, creative state, measurement source, billing exposure, and brand risk.",
        "workflows": [
            {
                "id": "campaign_pacing_shortfall",
                "workflow": "campaign_pacing",
                "title": "Campaign pacing shortfall workflow",
                "query": "A guaranteed campaign is underpacing before end of month; find makegood and inventory escalation guidance.",
                "expected_use": "attach pacing issue to delivery forecast, makegood policy, and sales communication",
                "metadata": {"event": "underpacing", "service": "campaign", "account_tier": "guaranteed"},
                "use_when": "campaign delivery is behind target or needs inventory/makegood decisions",
                "not_for": "brand safety blocks or creative rejection review",
            },
            {
                "id": "brand_safety_block",
                "workflow": "brand_safety",
                "title": "Brand safety block review",
                "query": "An advertiser's campaign was blocked on sensitive news content; find brand safety review steps.",
                "expected_use": "attach safety case to category review, override policy, and advertiser evidence",
                "metadata": {"event": "brand_block", "service": "inventory", "risk_signal": "sensitive_content"},
                "use_when": "inventory, content categories, or advertiser exclusions trigger brand safety decisions",
                "not_for": "campaign underdelivery or billing discrepancy reconciliation",
            },
            {
                "id": "creative_rejection",
                "workflow": "creative_approval",
                "title": "Creative rejection workflow",
                "query": "A video creative failed policy review for claims language; find creative approval and resubmission guidance.",
                "expected_use": "attach creative review to rejection reason, policy evidence, and resubmission workflow",
                "metadata": {"event": "creative_rejected", "service": "creative", "risk_signal": "claims_language"},
                "use_when": "creative assets require policy review, rejection handling, or resubmission",
                "not_for": "brand safety inventory blocks or measurement disputes",
            },
            {
                "id": "measurement_discrepancy",
                "workflow": "measurement_dispute",
                "title": "Measurement discrepancy workflow",
                "query": "Advertiser reports impression discrepancy between ad server and third-party measurement; find dispute steps.",
                "expected_use": "attach measurement case to reconciliation, threshold policy, and evidence exports",
                "metadata": {"event": "measurement_gap", "service": "ad_server", "risk_signal": "discrepancy"},
                "use_when": "reporting, impression, click, or conversion numbers differ across systems",
                "not_for": "creative rejection or campaign pacing shortfall",
            },
        ],
        "decoys": [
            {
                "id": "newsletter_sponsorship_copy",
                "summary": "Newsletter sponsorship copy note",
                "full": "Editorial sponsorship copy note unrelated to campaign pacing, brand safety, creative approval, or measurement disputes.",
                "workflow": "editorial",
                "kind": "content",
            }
        ],
    },
    {
        "id": "manufacturing_quality",
        "tenant": "forgewell_mfg",
        "label": "manufacturing quality operations",
        "seed_summary": "Manufacturing quality memory for nonconformance, supplier defects, calibration, line stoppage, and corrective action workflows.",
        "hub_workflow": "quality_ops",
        "hub_summary": "Manufacturing quality routing hub",
        "hub_full": "Routes manufacturing quality cases by defect source, product line, customer impact, containment state, and evidence requirements.",
        "workflows": [
            {
                "id": "supplier_nonconformance",
                "workflow": "supplier_quality",
                "title": "Supplier nonconformance workflow",
                "query": "Incoming parts from a supplier failed dimensional inspection; find nonconformance and containment steps.",
                "expected_use": "attach supplier defect to MRB review, containment, and corrective action evidence",
                "metadata": {"event": "incoming_defect", "entity": "supplier", "risk_signal": "nonconformance"},
                "use_when": "supplier parts fail inspection or require material review board disposition",
                "not_for": "tool calibration drift, safety lockout, or customer complaint intake",
            },
            {
                "id": "calibration_drift",
                "workflow": "equipment_calibration",
                "title": "Equipment calibration drift workflow",
                "query": "A torque tool was found out of calibration after production use; find impact review and quarantine guidance.",
                "expected_use": "attach calibration issue to affected lot review, tool quarantine, and QA approval",
                "metadata": {"event": "calibration_drift", "service": "equipment", "risk_signal": "quality_escape"},
                "use_when": "measurement or production tools are out of calibration and may affect product quality",
                "not_for": "supplier incoming defects or production line labor shortages",
            },
            {
                "id": "customer_quality_complaint",
                "workflow": "customer_complaint_quality",
                "title": "Customer quality complaint workflow",
                "query": "A customer reported field failures across multiple shipped units; find complaint triage and CAPA guidance.",
                "expected_use": "attach complaint to field failure analysis, CAPA, and customer response workflow",
                "metadata": {"event": "field_failure", "entity": "customer", "severity": "high"},
                "use_when": "customer complaints indicate shipped product defects or field failures",
                "not_for": "internal calibration checks or supplier-only receiving defects",
            },
            {
                "id": "line_stop_quality_hold",
                "workflow": "line_stop_hold",
                "title": "Line stop quality hold workflow",
                "query": "A production line was stopped after repeated assembly defects; find hold and release criteria.",
                "expected_use": "attach line stop to containment, root cause, and release approval steps",
                "metadata": {"event": "line_stop", "service": "assembly", "severity": "urgent"},
                "use_when": "production must stop due to repeated defects, safety risk, or unresolved quality escape",
                "not_for": "customer complaint follow-up or routine tool calibration schedule",
            },
        ],
        "decoys": [
            {
                "id": "cafeteria_supplier_menu",
                "summary": "Cafeteria supplier menu update",
                "full": "Facilities supplier note unrelated to manufacturing quality, calibration, customer complaints, or line holds.",
                "workflow": "facilities",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "banking_compliance_ops",
        "tenant": "stonebridge_bank",
        "label": "banking compliance operations",
        "seed_summary": "Banking operations memory for AML alerts, wire exceptions, loan documentation, customer complaints, and regulatory reporting.",
        "hub_workflow": "bank_compliance",
        "hub_summary": "Banking compliance routing hub",
        "hub_full": "Routes banking cases by account type, transaction signal, regulatory clock, evidence status, and approval ownership.",
        "workflows": [
            {
                "id": "aml_structuring_alert",
                "workflow": "aml_investigation",
                "title": "AML structuring alert investigation",
                "query": "A business account shows repeated cash deposits below reporting thresholds; find AML investigation guidance.",
                "expected_use": "attach alert to AML review, SAR decision evidence, and escalation policy",
                "metadata": {"risk_signal": "structuring", "entity": "business_account", "event": "cash_deposit"},
                "use_when": "transactions show AML typologies, suspicious patterns, or SAR review triggers",
                "not_for": "wire repair formatting, loan document exceptions, or service complaints",
            },
            {
                "id": "wire_exception_repair",
                "workflow": "wire_exception",
                "title": "Wire exception repair workflow",
                "query": "An outgoing wire failed beneficiary validation and needs repair before cutoff; find exception handling.",
                "expected_use": "attach wire issue to repair queue, approval, and customer communication workflow",
                "metadata": {"event": "wire_repair", "service": "payments", "severity": "high"},
                "use_when": "wire transfers fail validation, sanctions screening, cutoffs, or approval controls",
                "not_for": "AML structuring investigation or mortgage document review",
            },
            {
                "id": "loan_doc_exception",
                "workflow": "loan_documentation",
                "title": "Loan documentation exception workflow",
                "query": "A commercial loan is missing required guarantor documents before funding; find exception guidance.",
                "expected_use": "attach loan case to document checklist, exception approval, and funding hold policy",
                "metadata": {"entity": "borrower", "event": "missing_guarantor_doc", "service": "commercial_loan"},
                "use_when": "loan funding requires missing document resolution or approved exceptions",
                "not_for": "wire payment repair or customer complaint response",
            },
            {
                "id": "regulatory_complaint",
                "workflow": "regulatory_complaint_response",
                "title": "Regulatory complaint response workflow",
                "query": "A customer complaint arrived through a regulator portal; find response clock and evidence steps.",
                "expected_use": "attach complaint to regulatory deadline, investigation evidence, and response approval",
                "metadata": {"event": "regulator_complaint", "entity": "customer", "severity": "high"},
                "use_when": "complaints have regulatory deadlines, formal response obligations, or executive review",
                "not_for": "AML suspicious activity or loan funding checklist exceptions",
            },
        ],
        "decoys": [
            {
                "id": "branch_poster_refresh",
                "summary": "Branch poster refresh",
                "full": "Marketing branch poster update unrelated to AML, wires, loans, or regulatory complaints.",
                "workflow": "marketing",
                "kind": "marketing",
            }
        ],
    },
    {
        "id": "construction_project_ops",
        "tenant": "keystone_builders",
        "label": "construction project operations",
        "seed_summary": "Construction operations memory for RFIs, change orders, safety incidents, subcontractor delays, and inspection failures.",
        "hub_workflow": "construction_ops",
        "hub_summary": "Construction project routing hub",
        "hub_full": "Routes construction project cases by contract scope, schedule impact, safety severity, inspection status, and approval chain.",
        "workflows": [
            {
                "id": "change_order_scope",
                "workflow": "change_order_review",
                "title": "Change order scope review",
                "query": "A client requested extra electrical work that affects budget and schedule; find change order steps.",
                "expected_use": "attach scope change to pricing, schedule impact, and client approval workflow",
                "metadata": {"event": "scope_change", "service": "electrical", "risk_signal": "budget_schedule"},
                "use_when": "project scope, budget, or timeline changes require formal approval",
                "not_for": "safety incident investigation or failed inspection correction",
            },
            {
                "id": "safety_near_miss",
                "workflow": "safety_incident",
                "title": "Safety near-miss investigation workflow",
                "query": "A crane lift had a near miss with no injury but high severity; find safety investigation guidance.",
                "expected_use": "attach safety event to incident report, corrective action, and regulatory evidence",
                "metadata": {"event": "near_miss", "service": "crane", "severity": "high"},
                "use_when": "site safety events, near misses, injuries, or hazards require investigation",
                "not_for": "RFI clarification or client change order pricing",
            },
            {
                "id": "inspection_failure",
                "workflow": "inspection_correction",
                "title": "Inspection failure correction workflow",
                "query": "A city inspection failed due to fire-stopping issues; find correction and reinspection guidance.",
                "expected_use": "attach inspection failure to deficiency correction, photo evidence, and reinspection schedule",
                "metadata": {"event": "inspection_failed", "service": "fire_stopping", "jurisdiction": "city"},
                "use_when": "authority inspections fail and require correction, documentation, and reinspection",
                "not_for": "subcontractor delay notice or safety near-miss reporting",
            },
            {
                "id": "subcontractor_delay",
                "workflow": "schedule_delay",
                "title": "Subcontractor delay notice workflow",
                "query": "A subcontractor delay threatens the critical path; find notice and schedule mitigation steps.",
                "expected_use": "attach delay to notice requirements, mitigation plan, and schedule impact evidence",
                "metadata": {"event": "delay", "entity": "subcontractor", "risk_signal": "critical_path"},
                "use_when": "subcontractor or supplier delays affect critical path, milestones, or liquidated damages",
                "not_for": "inspection correction or client-requested scope change",
            },
        ],
        "decoys": [
            {
                "id": "site_lunch_vendor",
                "summary": "Site lunch vendor note",
                "full": "Administrative lunch vendor detail unrelated to construction change, safety, inspection, or delay workflows.",
                "workflow": "admin",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "telecom_network_ops",
        "tenant": "signalcrest_telco",
        "label": "telecommunications network operations",
        "seed_summary": "Telecom operations memory for fiber cuts, tower outages, provisioning failures, number porting, and SLA credits.",
        "hub_workflow": "network_ops",
        "hub_summary": "Telecom network routing hub",
        "hub_full": "Routes telecom cases by network layer, customer tier, outage scope, regulatory clock, and field dispatch requirements.",
        "workflows": [
            {
                "id": "fiber_cut_outage",
                "workflow": "fiber_outage_restoration",
                "title": "Fiber cut outage restoration",
                "query": "A backhoe cut fiber affecting enterprise circuits; find restoration and customer escalation guidance.",
                "expected_use": "attach outage to fiber restoration, field dispatch, and enterprise communication workflow",
                "metadata": {"event": "fiber_cut", "service": "enterprise_circuit", "severity": "critical"},
                "use_when": "fiber damage or transport outage affects customer circuits or network backbone",
                "not_for": "number port rejection or broadband provisioning workflow",
            },
            {
                "id": "tower_power_failure",
                "workflow": "wireless_site_outage",
                "title": "Tower power failure workflow",
                "query": "A cell tower lost power and backup battery is near depletion; find wireless site outage steps.",
                "expected_use": "attach site outage to power escalation, field dispatch, and customer impact reporting",
                "metadata": {"event": "power_loss", "service": "wireless_site", "severity": "high"},
                "use_when": "wireless sites lose power, backhaul, radio service, or environmental controls",
                "not_for": "fiber cut repair or number porting exceptions",
            },
            {
                "id": "provisioning_failure",
                "workflow": "service_provisioning",
                "title": "Service provisioning failure workflow",
                "query": "A business broadband order failed activation after equipment install; find provisioning repair guidance.",
                "expected_use": "attach activation failure to provisioning queue, field handoff, and customer notice",
                "metadata": {"event": "activation_failed", "service": "broadband", "entity": "business_customer"},
                "use_when": "new services fail activation, equipment install, or account provisioning",
                "not_for": "tower outage restoration or SLA credit dispute",
            },
            {
                "id": "number_port_rejection",
                "workflow": "number_porting",
                "title": "Number port rejection workflow",
                "query": "A customer's number port was rejected due to carrier mismatch; find correction and escalation steps.",
                "expected_use": "attach porting case to validation, losing-carrier evidence, and customer communication",
                "metadata": {"event": "port_rejected", "service": "number_porting", "entity": "customer"},
                "use_when": "phone number ports fail validation, carrier matching, or regulatory timing",
                "not_for": "fiber outage or broadband activation repair",
            },
        ],
        "decoys": [
            {
                "id": "retail_phone_display",
                "summary": "Retail phone display reset",
                "full": "Retail merchandising note unrelated to network outage, provisioning, or number porting operations.",
                "workflow": "retail_admin",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "food_safety_ops",
        "tenant": "freshfork_foods",
        "label": "food safety operations",
        "seed_summary": "Food safety memory for recalls, allergen controls, cold chain breaks, sanitation, and supplier audits.",
        "hub_workflow": "food_safety",
        "hub_summary": "Food safety routing hub",
        "hub_full": "Routes food safety cases by product lot, hazard type, supplier, temperature history, regulatory exposure, and evidence.",
        "workflows": [
            {
                "id": "allergen_mislabel",
                "workflow": "allergen_recall",
                "title": "Allergen mislabel recall workflow",
                "query": "A packaged product may omit a peanut allergen from the label; find recall and notification steps.",
                "expected_use": "attach allergen event to recall decision, lot trace, and regulator notification workflow",
                "metadata": {"event": "allergen_mislabel", "risk_signal": "peanut", "severity": "critical"},
                "use_when": "undeclared allergens, mislabels, or packaging errors may affect consumer safety",
                "not_for": "cold chain excursion or routine supplier audit",
            },
            {
                "id": "cold_chain_break",
                "workflow": "temperature_excursion",
                "title": "Cold chain temperature excursion workflow",
                "query": "A refrigerated shipment exceeded temperature limits for two hours; find disposition and evidence steps.",
                "expected_use": "attach temperature event to product hold, QA disposition, and carrier evidence",
                "metadata": {"event": "temperature_excursion", "service": "cold_chain", "severity": "high"},
                "use_when": "temperature-controlled food exceeds limits in storage, transit, or receiving",
                "not_for": "allergen mislabel recall or sanitation swab failure",
            },
            {
                "id": "sanitation_positive_swab",
                "workflow": "sanitation_corrective_action",
                "title": "Sanitation positive swab workflow",
                "query": "Environmental swab results are positive on a food contact surface; find corrective action guidance.",
                "expected_use": "attach swab result to sanitation hold, retest, and corrective action documentation",
                "metadata": {"event": "positive_swab", "service": "sanitation", "risk_signal": "pathogen"},
                "use_when": "environmental monitoring or sanitation checks find positive results",
                "not_for": "supplier audit scheduling or cold-chain carrier claims",
            },
            {
                "id": "supplier_audit_failure",
                "workflow": "supplier_food_safety",
                "title": "Supplier audit failure workflow",
                "query": "A supplier failed a food safety audit for incomplete traceability records; find supplier hold steps.",
                "expected_use": "attach audit failure to supplier hold, corrective plan, and approval evidence",
                "metadata": {"event": "audit_failed", "entity": "supplier", "risk_signal": "traceability"},
                "use_when": "supplier audits reveal food safety, traceability, or certification failures",
                "not_for": "consumer recall execution or sanitation positive swab response",
            },
        ],
        "decoys": [
            {
                "id": "recipe_photo_update",
                "summary": "Recipe photo update",
                "full": "Marketing photo update unrelated to food safety recall, temperature, sanitation, or supplier audit workflows.",
                "workflow": "marketing",
                "kind": "marketing",
            }
        ],
    },
    {
        "id": "property_management_ops",
        "tenant": "civitas_properties",
        "label": "property management operations",
        "seed_summary": "Property management memory for lease renewals, maintenance emergencies, tenant disputes, security deposits, and inspections.",
        "hub_workflow": "property_ops",
        "hub_summary": "Property management routing hub",
        "hub_full": "Routes property cases by tenant, unit, lease state, emergency severity, legal clock, and documentation evidence.",
        "workflows": [
            {
                "id": "emergency_leak_response",
                "workflow": "maintenance_emergency",
                "title": "Emergency leak response workflow",
                "query": "A tenant reports an active water leak affecting multiple units; find emergency maintenance guidance.",
                "expected_use": "attach leak case to emergency dispatch, tenant notification, and damage evidence",
                "metadata": {"event": "water_leak", "severity": "urgent", "entity": "tenant"},
                "use_when": "maintenance emergencies threaten habitability, property damage, or safety",
                "not_for": "lease renewal negotiation or deposit dispute response",
            },
            {
                "id": "lease_renewal_exception",
                "workflow": "lease_renewal",
                "title": "Lease renewal exception workflow",
                "query": "A tenant requests non-standard renewal terms after a rent increase notice; find renewal guidance.",
                "expected_use": "attach lease case to renewal approval, notice compliance, and tenant communication",
                "metadata": {"event": "renewal_exception", "entity": "tenant", "risk_signal": "rent_increase"},
                "use_when": "lease renewals, notices, concessions, or non-standard terms need review",
                "not_for": "emergency maintenance or security deposit itemization",
            },
            {
                "id": "security_deposit_dispute",
                "workflow": "deposit_dispute",
                "title": "Security deposit dispute workflow",
                "query": "A former tenant disputes cleaning and damage deductions; find deposit evidence and response steps.",
                "expected_use": "attach dispute to itemization, photo evidence, statutory timeline, and response approval",
                "metadata": {"event": "deposit_dispute", "entity": "tenant", "jurisdiction": "state"},
                "use_when": "move-out charges, deposit returns, or statutory response deadlines are disputed",
                "not_for": "active water leak dispatch or lease renewal approval",
            },
            {
                "id": "noise_complaint_escalation",
                "workflow": "tenant_dispute",
                "title": "Noise complaint escalation workflow",
                "query": "Repeated tenant noise complaints need documented escalation and lease enforcement review.",
                "expected_use": "attach complaint to documentation, warning notice, and lease enforcement workflow",
                "metadata": {"event": "noise_complaint", "entity": "tenant", "risk_signal": "repeat_issue"},
                "use_when": "tenant disputes, complaints, or lease violations require documented escalation",
                "not_for": "security deposit itemization or emergency maintenance dispatch",
            },
        ],
        "decoys": [
            {
                "id": "lobby_art_rotation",
                "summary": "Lobby art rotation",
                "full": "Amenities note for lobby art rotation unrelated to lease, maintenance, deposit, or tenant dispute workflows.",
                "workflow": "amenities",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "biotech_lab_ops",
        "tenant": "helixbridge_labs",
        "label": "biotech laboratory operations",
        "seed_summary": "Biotech lab memory for sample chain of custody, assay deviations, reagent lot failures, biosafety, and study documentation.",
        "hub_workflow": "lab_ops",
        "hub_summary": "Biotech lab operations routing hub",
        "hub_full": "Routes lab cases by study, sample, assay, biosafety level, reagent lot, and documentation control.",
        "workflows": [
            {
                "id": "sample_chain_gap",
                "workflow": "sample_chain_of_custody",
                "title": "Sample chain-of-custody gap workflow",
                "query": "A clinical sample has a missing custody timestamp before assay processing; find review and hold steps.",
                "expected_use": "attach sample issue to chain-of-custody review, sample hold, and study deviation evidence",
                "metadata": {"event": "custody_gap", "entity": "sample", "risk_signal": "study_integrity"},
                "use_when": "sample custody, identity, or accessioning gaps may affect study validity",
                "not_for": "reagent lot failure or biosafety exposure response",
            },
            {
                "id": "assay_deviation",
                "workflow": "assay_deviation",
                "title": "Assay deviation investigation workflow",
                "query": "An assay run exceeded control limits for a study batch; find deviation investigation guidance.",
                "expected_use": "attach assay issue to deviation report, rerun criteria, and QA approval",
                "metadata": {"event": "control_limit_failure", "service": "assay", "risk_signal": "invalid_run"},
                "use_when": "assay controls, acceptance criteria, or run conditions deviate from protocol",
                "not_for": "sample custody gaps or reagent quarantine intake",
            },
            {
                "id": "reagent_lot_failure",
                "workflow": "reagent_quality",
                "title": "Reagent lot failure workflow",
                "query": "A reagent lot failed QC and may have been used in previous runs; find quarantine and impact review.",
                "expected_use": "attach reagent issue to lot quarantine, affected run assessment, and supplier evidence",
                "metadata": {"event": "lot_qc_fail", "entity": "reagent", "risk_signal": "quality_escape"},
                "use_when": "reagent, kit, or material lots fail QC or may affect previous experiments",
                "not_for": "biosafety exposure or sample custody documentation",
            },
            {
                "id": "biosafety_exposure",
                "workflow": "biosafety_incident",
                "title": "Biosafety exposure workflow",
                "query": "A lab worker reported possible exposure during BSL-2 handling; find biosafety incident steps.",
                "expected_use": "attach exposure to incident report, medical evaluation, and biosafety notification",
                "metadata": {"event": "possible_exposure", "service": "bsl2", "severity": "high"},
                "use_when": "biosafety, exposure, spill, or containment incidents require immediate response",
                "not_for": "assay deviation investigation or reagent lot quarantine",
            },
        ],
        "decoys": [
            {
                "id": "lab_coffee_machine",
                "summary": "Lab coffee machine repair",
                "full": "Facilities repair ticket unrelated to sample custody, assay deviations, reagent quality, or biosafety response.",
                "workflow": "facilities",
                "kind": "admin",
            }
        ],
    },
    {
        "id": "nonprofit_grants_ops",
        "tenant": "brightpath_foundation",
        "label": "nonprofit grants operations",
        "seed_summary": "Nonprofit operations memory for grant compliance, donor restrictions, program reporting, volunteer screening, and expense allowability.",
        "hub_workflow": "nonprofit_ops",
        "hub_summary": "Nonprofit grants routing hub",
        "hub_full": "Routes nonprofit cases by donor, restriction, program, reporting deadline, expense allowability, and compliance evidence.",
        "workflows": [
            {
                "id": "restricted_grant_expense",
                "workflow": "grant_expense_allowability",
                "title": "Restricted grant expense allowability workflow",
                "query": "A program team wants to charge travel costs to a restricted grant; find allowability review guidance.",
                "expected_use": "attach expense request to donor restriction, budget evidence, and approval workflow",
                "metadata": {"event": "restricted_expense", "entity": "grant", "risk_signal": "allowability"},
                "use_when": "expenses must be checked against grant budgets, donor restrictions, or allowability rules",
                "not_for": "volunteer screening or program outcome report drafting",
            },
            {
                "id": "donor_restriction_change",
                "workflow": "donor_restriction",
                "title": "Donor restriction change workflow",
                "query": "A donor approved a restriction change by email and finance needs documentation steps.",
                "expected_use": "attach restriction change to donor evidence, finance update, and board reporting",
                "metadata": {"event": "restriction_change", "entity": "donor", "service": "finance"},
                "use_when": "donor-imposed restrictions, releases, or designation changes affect fund use",
                "not_for": "grant expense allowability or volunteer background check",
            },
            {
                "id": "program_outcome_report",
                "workflow": "grant_reporting",
                "title": "Program outcome reporting workflow",
                "query": "A funder report is due with beneficiary counts and outcome metrics; find reporting evidence guidance.",
                "expected_use": "attach reporting task to metric validation, narrative approval, and deadline controls",
                "metadata": {"event": "report_due", "entity": "funder", "service": "program"},
                "use_when": "grant reports require outcome metrics, narratives, financials, or funder deadlines",
                "not_for": "donor restriction release or volunteer screening intake",
            },
            {
                "id": "volunteer_screening",
                "workflow": "volunteer_compliance",
                "title": "Volunteer screening workflow",
                "query": "A volunteer will work with minors and needs background screening before an event; find compliance steps.",
                "expected_use": "attach volunteer case to screening, consent evidence, and event eligibility workflow",
                "metadata": {"event": "minor_contact", "entity": "volunteer", "risk_signal": "screening"},
                "use_when": "volunteers require background checks, consent, training, or eligibility review",
                "not_for": "grant reporting or restricted expense review",
            },
        ],
        "decoys": [
            {
                "id": "newsletter_thank_you",
                "summary": "Donor newsletter thank-you copy",
                "full": "Communications copy unrelated to grant restrictions, reporting, expense allowability, or volunteer compliance.",
                "workflow": "communications",
                "kind": "marketing",
            }
        ],
    },
)


if __name__ == "__main__":
    main()
