#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vector_graph import embed_text
from vector_graph.vectors import effective_summary_vector, metadata_vector_from, stable_edge_vector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a manually curated domain-validation benchmark without teacher labels."
    )
    parser.add_argument("--output-dir", default="data/domain_validation_curated")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    write_cases(cases, output_dir=output_dir)
    print(
        f"wrote {len(cases)} traversal ranking and {len(cases)} attach ranking cases "
        f"to {output_dir}"
    )


def build_cases() -> list[dict]:
    domains = domain_pack()
    cases = []
    for domain in domains:
        for query in domain["queries"]:
            cases.append(build_case(domain, query))
    return cases


def build_case(domain: dict, query: dict) -> dict:
    node_by_id = {node["id"]: node for node in domain["nodes"]}
    current_node = {
        "summary": domain["seed_summary"],
        "metadata": {
            "domain": domain["id"],
            "tenant": domain["tenant"],
            "kind": "seed",
            "workflow": query["workflow"],
        },
    }
    query_metadata = {
        "domain": domain["id"],
        "tenant": domain["tenant"],
        "workflow": query["workflow"],
        **query.get("metadata", {}),
    }
    query_vector = effective_text_vector(query["text"], query_metadata, 32)
    current_summary = effective_text_vector(current_node["summary"], current_node["metadata"], 32)
    path_vector = effective_text_vector(
        f"{domain['id']} {query['workflow']} {query.get('path_hint', '')}",
        {"domain": domain["id"], "tenant": domain["tenant"], "workflow": query["workflow"]},
        32,
    )
    new_full = embed_text(f"{query['text']} {query['expected_use']}", 64)
    positives = set(query["positive_ids"])
    candidate_ids = ordered_candidate_ids(domain, query)

    traversal_candidates = []
    attach_candidates = []
    for candidate_id in candidate_ids:
        node = node_by_id[candidate_id]
        is_positive = candidate_id in positives
        target = node_target(node, query, is_positive=is_positive)
        summary = effective_text_vector(node["summary"], node["metadata"], 32)
        full = embed_text(node["full"], 64)
        kind = candidate_kind(node, is_positive=is_positive, is_bridge=candidate_id in query.get("bridge_ids", []))
        weight = 1.5 if "hard" in kind or "negative" in kind else 1.0
        traversal_candidates.append(
            {
                "id": f"{domain['id']}:{query['id']}:{candidate_id}",
                "kind": kind,
                "label": 1 if target["follow"] >= 0.65 else 0,
                "rank_target": round(target["follow"], 6),
                "result_label": 1 if target["result"] >= 0.65 else 0,
                "result_rank_target": round(target["result"], 6),
                "weight": weight,
                "dst_summary": round_vector(summary),
                "edge": round_vector(stable_edge_vector(current_summary, summary, 16)),
                "confidence": node.get("confidence", 0.82),
                "hop": node.get("hop", 1),
                "oracle": round_vector(
                    [
                        target["follow"],
                        target["read_full"],
                        target["include"],
                        target["expand"],
                        target["stop"],
                        target["result"],
                    ]
                ),
            }
        )
        attach_candidates.append(
            {
                "id": f"{domain['id']}:{query['id']}:{candidate_id}:attach",
                "kind": kind,
                "label": 1 if target["include"] >= 0.65 else 0,
                "rank_target": round(target["include"], 6),
                "weight": weight,
                "candidate_summary": round_vector(summary),
                "candidate_full": round_vector(full),
                "oracle": round(target["include"], 6),
            }
        )

    return {
        "id": f"{domain['id']}:{query['id']}",
        "domain": domain["id"],
        "query_text": query["text"],
        "traversal": {
            "schema_version": 1,
            "kind": "domain_validation_traversal_ranking",
            "id": f"{domain['id']}:{query['id']}:traversal-ranking",
            "query": round_vector(query_vector),
            "current_summary": round_vector(current_summary),
            "path": round_vector(path_vector),
            "candidates": traversal_candidates,
        },
        "attach": {
            "schema_version": 1,
            "kind": "domain_validation_attach_ranking",
            "id": f"{domain['id']}:{query['id']}:attach-ranking",
            "new_summary": round_vector(query_vector),
            "new_full": round_vector(new_full),
            "path": round_vector(path_vector),
            "candidates": attach_candidates,
        },
    }


def node_target(node: dict, query: dict, *, is_positive: bool) -> dict[str, float]:
    if is_positive:
        read_full = 0.96 if node.get("requires_full_read", True) else 0.72
        return {
            "follow": 0.94,
            "read_full": read_full,
            "include": 0.96,
            "expand": 0.45,
            "stop": 0.03,
            "result": 0.97,
        }
    if node["kind"] == "bridge_positive":
        return {
            "follow": 0.82,
            "read_full": 0.28,
            "include": 0.22,
            "expand": 0.93,
            "stop": 0.08,
            "result": 0.18,
        }
    if "hard" in node["kind"]:
        return {
            "follow": 0.18,
            "read_full": 0.62,
            "include": 0.08,
            "expand": 0.16,
            "stop": 0.72,
            "result": 0.06,
        }
    return {
        "follow": 0.05,
        "read_full": 0.18,
        "include": 0.02,
        "expand": 0.05,
        "stop": 0.86,
        "result": 0.02,
    }


def candidate_kind(node: dict, *, is_positive: bool, is_bridge: bool) -> str:
    if is_positive:
        return "positive"
    if is_bridge:
        return "bridge_positive"
    if node["kind"] in {"positive", "bridge_positive"}:
        return "hard_same_domain_negative"
    return node["kind"]


def ordered_candidate_ids(domain: dict, query: dict) -> list[str]:
    selected = []
    for candidate_id in query["positive_ids"] + query.get("bridge_ids", []) + query.get("hard_negative_ids", []):
        if candidate_id not in selected:
            selected.append(candidate_id)
    for node in domain["nodes"]:
        if node["id"] not in selected:
            selected.append(node["id"])
    return selected[: query.get("candidate_count", 12)]


def write_cases(cases: Sequence[dict], *, output_dir: Path) -> None:
    traversal_path = output_dir / "traversal_ranking.jsonl"
    attach_path = output_dir / "attach_ranking.jsonl"
    manifest_path = output_dir / "manifest.json"
    with traversal_path.open("w", encoding="utf-8") as traversal_output, attach_path.open(
        "w", encoding="utf-8"
    ) as attach_output:
        for case in cases:
            traversal_output.write(json.dumps(case["traversal"], separators=(",", ":")) + "\n")
            attach_output.write(json.dumps(case["attach"], separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": 1,
        "name": "domain_validation_curated",
        "description": (
            "Hand-authored business-domain validation cases. Labels are manually assigned; "
            "no Qwen teacher scores are used as targets."
        ),
        "traversal_ranking": traversal_path.name,
        "attach_ranking": attach_path.name,
        "traversal_ranking_cases": len(cases),
        "attach_ranking_cases": len(cases),
        "dimensions": {"query": 32, "summary": 32, "edge": 16, "path": 32, "full": 64},
        "domains": sorted({case["domain"] for case in cases}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def effective_text_vector(text: str, metadata: Mapping[str, object], dimension: int) -> list[float]:
    return list(
        effective_summary_vector(
            embed_text(text, dimension),
            metadata_vector_from(metadata, dimension),
            dimension,
        )
    )


def round_vector(values: Iterable[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def domain_pack() -> list[dict]:
    return [
        {
            "id": "saas_billing_ops",
            "tenant": "northwind_saas",
            "seed_summary": "Billing operations memory for enterprise SaaS accounts, invoices, usage, refunds, and dunning.",
            "queries": [
                {
                    "id": "failed_invoice_refund",
                    "workflow": "refund_escalation",
                    "text": "Enterprise customer Acme was double charged after a failed annual invoice retry; find refund and ledger handling.",
                    "expected_use": "attach a refund incident to billing, ledger reconciliation, and enterprise invoice policy",
                    "metadata": {"account_tier": "enterprise", "payment_flow": "invoice_retry"},
                    "positive_ids": ["refund_retry_policy", "ledger_reversal_runbook", "acme_contract_terms"],
                    "bridge_ids": ["billing_refunds_hub"],
                    "hard_negative_ids": ["self_serve_refund_policy", "usage_overage_notice"],
                },
                {
                    "id": "usage_overage_notice",
                    "workflow": "usage_billing",
                    "text": "Prepare the correct customer notice for workspace usage overage before the next invoice closes.",
                    "expected_use": "attach usage overage record to notification policy and invoice preview workflow",
                    "metadata": {"account_tier": "business", "billing_event": "usage_overage"},
                    "positive_ids": ["usage_overage_notice", "invoice_preview_timeline"],
                    "bridge_ids": ["billing_refunds_hub"],
                    "hard_negative_ids": ["refund_retry_policy", "acme_contract_terms"],
                },
            ],
            "nodes": [
                node("billing_refunds_hub", "Refunds and billing adjustments routing hub", "Index node for refund, ledger, invoice, and contract adjustment workflows.", {"kind": "hub", "workflow": "refund_escalation"}, "bridge_positive", confidence=0.92, hop=0, requires_full_read=False),
                node("refund_retry_policy", "Enterprise invoice retry refund policy", "When an invoice retry succeeds after a gateway failure, verify duplicate capture, issue account credit or payment refund, and notify finance.", {"kind": "policy", "account_tier": "enterprise", "workflow": "refund_escalation"}, "positive"),
                node("ledger_reversal_runbook", "Ledger reversal runbook for duplicate charges", "Finance must post a reversal entry, link the original payment intent, and reconcile settlement totals before closing the incident.", {"kind": "runbook", "system": "ledger", "workflow": "refund_escalation"}, "positive"),
                node("acme_contract_terms", "Acme annual contract billing terms", "Acme has annual invoicing with manual refund approval above $5,000 and a two-business-day incident response SLA.", {"kind": "contract", "customer": "Acme", "account_tier": "enterprise"}, "positive"),
                node("self_serve_refund_policy", "Self-serve card refund policy", "Self-serve customers can request card refunds for monthly subscriptions below $500 without finance approval.", {"kind": "policy", "account_tier": "self_serve", "workflow": "refund"}, "hard_same_domain_negative"),
                node("usage_overage_notice", "Workspace usage overage customer notice", "Send usage overage notices before invoice close when workspace consumption crosses the committed plan allowance.", {"kind": "policy", "workflow": "usage_billing"}, "positive"),
                node("invoice_preview_timeline", "Invoice preview and close timeline", "Invoice previews are generated five days before billing close, with customer-visible usage line items and dispute links.", {"kind": "runbook", "workflow": "usage_billing"}, "positive"),
                node("chargeback_dispute_playbook", "Credit card chargeback dispute playbook", "Collect evidence and subscription logs for card chargebacks; do not use for invoice retry duplicate captures.", {"kind": "runbook", "workflow": "chargeback"}, "hard_same_domain_negative"),
                node("tax_exemption_certificate", "Tax exemption certificate validation", "Validate tax certificates before invoice issue and store jurisdiction-specific evidence.", {"kind": "compliance", "workflow": "tax"}, "hard_compliance_negative"),
                node("support_password_reset", "Support password reset macro", "Authenticate user identity and send password reset instructions.", {"kind": "support", "workflow": "identity"}, "cross_domain_negative"),
                node("sales_discount_request", "Sales discount approval memory", "Discount approvals require regional sales manager signoff.", {"kind": "sales", "workflow": "discount"}, "cross_domain_negative"),
                node("deployment_freeze_calendar", "Deployment freeze calendar", "Production deployment freezes around quarter close.", {"kind": "engineering", "workflow": "release"}, "cross_domain_negative"),
            ],
        },
        {
            "id": "fintech_risk",
            "tenant": "riverbank_payments",
            "seed_summary": "Risk operations memory for fintech onboarding, transaction monitoring, sanctions, KYC, disputes, and fraud review.",
            "queries": [
                {
                    "id": "merchant_velocity_hold",
                    "workflow": "merchant_risk_review",
                    "text": "A new merchant has abnormal payout velocity and mismatched business category; find the correct hold and enhanced review guidance.",
                    "expected_use": "attach merchant risk alert to velocity hold policy and enhanced KYC review",
                    "metadata": {"risk_signal": "velocity", "entity": "merchant"},
                    "positive_ids": ["merchant_velocity_hold", "enhanced_kyc_review", "payout_release_controls"],
                    "bridge_ids": ["risk_review_hub"],
                    "hard_negative_ids": ["consumer_card_velocity", "low_risk_auto_approval"],
                },
                {
                    "id": "sanctions_name_match",
                    "workflow": "sanctions_review",
                    "text": "A beneficial owner has a fuzzy sanctions name match; find escalation and evidence retention rules.",
                    "expected_use": "attach sanctions review to escalation workflow and evidence policy",
                    "metadata": {"risk_signal": "sanctions", "entity": "beneficial_owner"},
                    "positive_ids": ["sanctions_escalation", "beneficial_owner_evidence"],
                    "bridge_ids": ["risk_review_hub"],
                    "hard_negative_ids": ["merchant_velocity_hold", "consumer_card_velocity"],
                },
            ],
            "nodes": [
                node("risk_review_hub", "Risk review routing hub", "Routes merchant, consumer, sanctions, and payout risk reviews by entity and signal.", {"kind": "hub", "workflow": "risk"}, "bridge_positive", confidence=0.94, hop=0, requires_full_read=False),
                node("merchant_velocity_hold", "Merchant payout velocity hold policy", "Place a temporary payout hold when new merchant volume exceeds expected category velocity and ownership checks are incomplete.", {"kind": "policy", "workflow": "merchant_risk_review", "entity": "merchant"}, "positive"),
                node("enhanced_kyc_review", "Enhanced KYC review for mismatched business category", "Request ownership documents, business category evidence, and processor history before clearing merchant payout risk.", {"kind": "runbook", "workflow": "merchant_risk_review", "entity": "merchant"}, "positive"),
                node("payout_release_controls", "Payout release controls after risk hold", "Release payouts only after risk owner approval, case notes, and monitoring thresholds are updated.", {"kind": "control", "workflow": "merchant_risk_review"}, "positive"),
                node("consumer_card_velocity", "Consumer card velocity rule", "Consumer card velocity alerts trigger cardholder authentication checks and do not control merchant payout holds.", {"kind": "policy", "workflow": "consumer_fraud", "entity": "cardholder"}, "hard_same_domain_negative"),
                node("low_risk_auto_approval", "Low risk merchant auto approval", "Auto-approve merchants with verified ownership, low initial volume, and no category mismatch.", {"kind": "policy", "workflow": "merchant_onboarding"}, "hard_same_domain_negative"),
                node("sanctions_escalation", "Sanctions fuzzy match escalation", "Escalate fuzzy sanctions matches for beneficial owners to compliance review before account activation or payout release.", {"kind": "compliance", "workflow": "sanctions_review", "entity": "beneficial_owner"}, "positive"),
                node("beneficial_owner_evidence", "Beneficial owner evidence retention", "Retain identity documents, screening output, analyst notes, and final disposition for the required compliance period.", {"kind": "evidence", "workflow": "sanctions_review"}, "positive"),
                node("dispute_chargeback_packet", "Dispute chargeback evidence packet", "Compile fulfillment, authorization, and communication evidence for payment disputes.", {"kind": "dispute", "workflow": "chargeback"}, "hard_same_domain_negative"),
                node("support_invoice_macro", "Support invoice copy macro", "Send customers a copy of a billing invoice.", {"kind": "support", "workflow": "billing"}, "cross_domain_negative"),
                node("warehouse_shortage_alert", "Warehouse shortage alert", "Inventory shortage alert for replenishment planning.", {"kind": "supply_chain", "workflow": "inventory"}, "cross_domain_negative"),
                node("kubernetes_restart_runbook", "Kubernetes restart runbook", "Restart a failed deployment after checking readiness probes.", {"kind": "engineering", "workflow": "incident"}, "cross_domain_negative"),
            ],
        },
        {
            "id": "healthcare_ops",
            "tenant": "clearclinic",
            "seed_summary": "Clinic operations memory for referrals, prior authorization, lab follow-up, patient safety, and compliance.",
            "queries": [
                {
                    "id": "prior_auth_denial",
                    "workflow": "prior_authorization",
                    "text": "A specialist referral was denied for missing prior authorization documentation; find appeal packet requirements.",
                    "expected_use": "attach referral denial to prior authorization appeal and payer evidence checklist",
                    "metadata": {"payer": "commercial", "care_flow": "specialist_referral"},
                    "positive_ids": ["prior_auth_appeal_packet", "payer_evidence_checklist"],
                    "bridge_ids": ["clinical_admin_hub"],
                    "hard_negative_ids": ["lab_result_critical_callback", "hipaa_release_form"],
                },
                {
                    "id": "critical_lab_callback",
                    "workflow": "critical_lab_followup",
                    "text": "A critical potassium lab result needs documented callback and escalation if the patient is unreachable.",
                    "expected_use": "attach critical lab to callback workflow and escalation documentation",
                    "metadata": {"safety": "critical_result", "care_flow": "lab_followup"},
                    "positive_ids": ["lab_result_critical_callback", "unreachable_patient_escalation"],
                    "bridge_ids": ["clinical_admin_hub"],
                    "hard_negative_ids": ["prior_auth_appeal_packet", "routine_lab_portal_message"],
                },
            ],
            "nodes": [
                node("clinical_admin_hub", "Clinical administration routing hub", "Routes referral, prior authorization, lab follow-up, records release, and safety workflows.", {"kind": "hub", "workflow": "clinical_admin"}, "bridge_positive", confidence=0.93, hop=0, requires_full_read=False),
                node("prior_auth_appeal_packet", "Prior authorization appeal packet", "Appeal packets include referral order, clinical notes, diagnosis codes, payer denial reason, and requested service dates.", {"kind": "runbook", "workflow": "prior_authorization"}, "positive"),
                node("payer_evidence_checklist", "Payer evidence checklist for specialist referrals", "Collect medical necessity evidence, previous conservative treatment, and provider attestation for specialist referral review.", {"kind": "checklist", "workflow": "prior_authorization"}, "positive"),
                node("lab_result_critical_callback", "Critical lab result callback workflow", "Critical lab results require direct callback documentation, clinician notification, and timestamped acknowledgement.", {"kind": "safety", "workflow": "critical_lab_followup"}, "positive"),
                node("unreachable_patient_escalation", "Unreachable patient escalation for critical results", "If the patient cannot be reached, escalate to clinician, alternate contact, and emergency protocol according to severity.", {"kind": "safety", "workflow": "critical_lab_followup"}, "positive"),
                node("routine_lab_portal_message", "Routine lab portal message template", "Non-critical normal lab results may be released through the patient portal with standard educational text.", {"kind": "template", "workflow": "routine_lab"}, "hard_same_domain_negative"),
                node("hipaa_release_form", "HIPAA records release form", "Records release requires signed authorization and identity verification; unrelated to clinical payer appeal content.", {"kind": "compliance", "workflow": "records_release"}, "hard_compliance_negative"),
                node("appointment_reminder_sms", "Appointment reminder SMS", "Send automated reminders before scheduled appointments.", {"kind": "template", "workflow": "scheduling"}, "cross_domain_negative"),
                node("merchant_velocity_hold", "Merchant payout velocity hold policy", "Place a payout hold when merchant volume is abnormal.", {"kind": "fintech", "workflow": "risk"}, "cross_domain_negative"),
                node("invoice_preview_timeline", "Invoice preview timeline", "Invoice previews are generated before billing close.", {"kind": "billing", "workflow": "invoice"}, "cross_domain_negative"),
                node("incident_postmortem_template", "Engineering postmortem template", "Document incident timeline, contributing factors, and action items.", {"kind": "engineering", "workflow": "incident"}, "cross_domain_negative"),
                node("inventory_cycle_count", "Inventory cycle count procedure", "Count high-value warehouse inventory by location.", {"kind": "supply_chain", "workflow": "inventory"}, "cross_domain_negative"),
            ],
        },
        {
            "id": "supply_chain",
            "tenant": "atlas_distribution",
            "seed_summary": "Distribution operations memory for purchase orders, inventory, receiving exceptions, vendor chargebacks, and fulfillment.",
            "queries": [
                {
                    "id": "late_po_short_ship",
                    "workflow": "receiving_exception",
                    "text": "A vendor purchase order arrived late and short shipped two SKUs; find receiving exception and vendor chargeback steps.",
                    "expected_use": "attach receiving discrepancy to shortage workflow and vendor chargeback policy",
                    "metadata": {"event": "short_ship", "actor": "vendor"},
                    "positive_ids": ["receiving_short_ship_exception", "vendor_chargeback_policy", "po_delay_escalation"],
                    "bridge_ids": ["warehouse_ops_hub"],
                    "hard_negative_ids": ["customer_return_restock", "inventory_cycle_count"],
                },
                {
                    "id": "inventory_recount",
                    "workflow": "inventory_accuracy",
                    "text": "Cycle count found high-value SKU variance across two warehouse bins; find recount and adjustment approval rules.",
                    "expected_use": "attach count variance to cycle count recount and inventory adjustment approval",
                    "metadata": {"event": "count_variance", "inventory_class": "high_value"},
                    "positive_ids": ["inventory_cycle_count", "inventory_adjustment_approval"],
                    "bridge_ids": ["warehouse_ops_hub"],
                    "hard_negative_ids": ["receiving_short_ship_exception", "customer_return_restock"],
                },
            ],
            "nodes": [
                node("warehouse_ops_hub", "Warehouse operations routing hub", "Routes receiving, inventory accuracy, returns, vendor chargeback, and fulfillment exception workflows.", {"kind": "hub", "workflow": "warehouse_ops"}, "bridge_positive", confidence=0.91, hop=0, requires_full_read=False),
                node("receiving_short_ship_exception", "Receiving short-ship exception workflow", "Log missing SKUs, capture packing slip evidence, quarantine partial receipt, and notify procurement for vendor follow-up.", {"kind": "runbook", "workflow": "receiving_exception"}, "positive"),
                node("vendor_chargeback_policy", "Vendor chargeback policy for shortages", "Chargebacks require PO, ASN, receiving photos, discrepancy count, and buyer approval before debit memo issue.", {"kind": "policy", "workflow": "vendor_chargeback"}, "positive"),
                node("po_delay_escalation", "Purchase order delay escalation", "Late vendor arrivals trigger buyer notification, revised ETA capture, and customer allocation review.", {"kind": "runbook", "workflow": "receiving_exception"}, "positive"),
                node("customer_return_restock", "Customer return restock workflow", "Returned customer units are inspected and restocked or scrapped; do not use for inbound vendor shortages.", {"kind": "runbook", "workflow": "returns"}, "hard_same_domain_negative"),
                node("inventory_cycle_count", "High-value inventory cycle count procedure", "High-value SKUs require blind recount, supervisor verification, and variance notes by bin location.", {"kind": "procedure", "workflow": "inventory_accuracy"}, "positive"),
                node("inventory_adjustment_approval", "Inventory adjustment approval rules", "Inventory adjustments above tolerance require operations manager approval and audit reason codes.", {"kind": "control", "workflow": "inventory_accuracy"}, "positive"),
                node("carrier_late_delivery_claim", "Carrier late delivery claim", "Carrier claims require tracking evidence and customer delivery timestamps.", {"kind": "policy", "workflow": "carrier_claim"}, "hard_same_domain_negative"),
                node("refund_retry_policy", "Enterprise invoice retry refund policy", "Verify duplicate capture and issue refund.", {"kind": "billing", "workflow": "refund"}, "cross_domain_negative"),
                node("sanctions_escalation", "Sanctions escalation workflow", "Escalate sanctions fuzzy matches.", {"kind": "fintech", "workflow": "sanctions"}, "cross_domain_negative"),
                node("critical_lab_callback", "Critical lab callback workflow", "Notify clinician for critical lab results.", {"kind": "healthcare", "workflow": "critical_lab"}, "cross_domain_negative"),
                node("slo_burn_rate_alert", "SLO burn rate alert runbook", "Investigate service-level burn rate alerts.", {"kind": "engineering", "workflow": "incident"}, "cross_domain_negative"),
            ],
        },
        {
            "id": "devops_incident",
            "tenant": "orbit_cloud",
            "seed_summary": "Cloud operations memory for incidents, deployments, observability, rollback, data migrations, and customer comms.",
            "queries": [
                {
                    "id": "api_latency_burn",
                    "workflow": "production_incident",
                    "text": "API latency SLO burn started after a deploy; find rollback, incident comms, and trace investigation guidance.",
                    "expected_use": "attach latency incident to SLO burn runbook, rollback policy, and status page communication",
                    "metadata": {"service": "api", "signal": "latency", "event": "deploy"},
                    "positive_ids": ["slo_burn_rate_alert", "deploy_rollback_policy", "status_page_update"],
                    "bridge_ids": ["incident_response_hub"],
                    "hard_negative_ids": ["database_migration_plan", "batch_job_delay"],
                },
                {
                    "id": "migration_lock_wait",
                    "workflow": "database_migration",
                    "text": "A production database migration is causing lock waits; find pause, rollback, and customer impact procedure.",
                    "expected_use": "attach migration issue to migration rollback and incident communication guidance",
                    "metadata": {"service": "database", "signal": "lock_wait", "event": "migration"},
                    "positive_ids": ["database_migration_plan", "migration_rollback_runbook", "status_page_update"],
                    "bridge_ids": ["incident_response_hub"],
                    "hard_negative_ids": ["deploy_rollback_policy", "batch_job_delay"],
                },
            ],
            "nodes": [
                node("incident_response_hub", "Incident response routing hub", "Routes production incidents by service, deploy, migration, customer communication, and observability signal.", {"kind": "hub", "workflow": "incident_response"}, "bridge_positive", confidence=0.94, hop=0, requires_full_read=False),
                node("slo_burn_rate_alert", "API SLO burn rate alert runbook", "For API latency burn, check traces, recent deploys, regional saturation, and error budget impact before mitigation.", {"kind": "runbook", "workflow": "production_incident", "service": "api"}, "positive"),
                node("deploy_rollback_policy", "Production deploy rollback policy", "Rollback deploys when error budget burn or latency regression is tied to a recent release and mitigation is not immediate.", {"kind": "policy", "workflow": "production_incident", "event": "deploy"}, "positive"),
                node("status_page_update", "Customer status page update policy", "Post customer-facing status updates when production impact exceeds threshold or incident commander declares degraded service.", {"kind": "comms", "workflow": "incident_response"}, "positive"),
                node("database_migration_plan", "Production database migration plan", "Migrations require lock monitoring, pause criteria, validation queries, and rollback ownership.", {"kind": "plan", "workflow": "database_migration"}, "positive"),
                node("migration_rollback_runbook", "Database migration rollback runbook", "Pause writes if safe, revert migration steps, validate schema state, and communicate customer impact.", {"kind": "runbook", "workflow": "database_migration"}, "positive"),
                node("batch_job_delay", "Batch job delay alert", "Batch job delays are handled by queue scaling and retry backoff, not deploy rollback.", {"kind": "runbook", "workflow": "batch_processing"}, "hard_same_domain_negative"),
                node("feature_flag_cleanup", "Feature flag cleanup task", "Remove expired feature flags after rollout completion.", {"kind": "maintenance", "workflow": "cleanup"}, "cross_domain_negative"),
                node("vendor_chargeback_policy", "Vendor chargeback policy", "Chargebacks require PO and receiving evidence.", {"kind": "supply_chain", "workflow": "vendor_chargeback"}, "cross_domain_negative"),
                node("prior_auth_appeal_packet", "Prior authorization appeal packet", "Appeal packet for specialist referral denial.", {"kind": "healthcare", "workflow": "prior_authorization"}, "cross_domain_negative"),
                node("merchant_velocity_hold", "Merchant velocity hold", "Merchant payout hold policy.", {"kind": "fintech", "workflow": "risk"}, "cross_domain_negative"),
                node("tax_exemption_certificate", "Tax exemption validation", "Validate tax exemption certificate.", {"kind": "billing", "workflow": "tax"}, "cross_domain_negative"),
            ],
        },
    ]


def node(
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


if __name__ == "__main__":
    main()
