import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


YES = {"1", "true", "yes", "y", "approved", "verified", "pass"}
NO = {"0", "false", "no", "n", "rejected", "fail"}
UNKNOWN = {"", "unknown", "pending", "not_checked", "not checked", "n/a"}


def split_counts(sample_count):
    if sample_count <= 0:
        return {"train": 0, "validation": 0, "test": 0}, "no_video_reference"
    if sample_count <= 3:
        return {"train": sample_count, "validation": 0, "test": 0}, "train_only"
    if sample_count <= 5:
        return {"train": sample_count - 1, "validation": 1, "test": 0}, "train_validation_only"
    test_count = max(1, round(sample_count * 0.15))
    validation_count = max(1, round(sample_count * 0.15))
    return {
        "train": sample_count - validation_count - test_count,
        "validation": validation_count,
        "test": test_count,
    }, "train_validation_test"


def audit_priority(sample_count):
    if sample_count >= 6:
        return "P1"
    if sample_count >= 4:
        return "P2"
    if sample_count >= 2:
        return "P3"
    return "P4"


def is_yes(value):
    return (value or "").strip().casefold() in YES


def final_eligibility(row):
    if not row.get("video_id"):
        return False, "missing_video_reference"
    if not is_yes(row.get("video_accessible")):
        return False, "video_access_not_confirmed"
    if not is_yes(row.get("is_vsl")):
        return False, "vsl_content_not_confirmed"
    if not is_yes(row.get("sign_matches_label")):
        return False, "label_not_confirmed"
    if (row.get("visual_quality") or "").strip().casefold() not in {"good", "usable", "acceptable", "clear", "pass"}:
        return False, "visual_quality_not_approved"
    if not is_yes(row.get("concept_review_status")):
        return False, "concept_not_approved"
    if not is_yes(row.get("training_eligible")):
        return False, "training_eligibility_not_approved"
    if row.get("is_duplicate_content", "").strip().casefold() not in NO:
        return False, "duplicate_status_not_cleared"
    license_status = (row.get("license_status") or "").strip().casefold()
    if license_status in UNKNOWN or license_status in {"denied", "not permitted", "not_permitted"}:
        return False, "license_not_cleared"
    return True, "approved"


def stable_order(rows, seed, label_key):
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}|{label_key}|{row.get('source_id')}|{row.get('video_id')}".encode("utf-8")).hexdigest(),
            row.get("sample_id", ""),
        ),
    )


def assign_group_splits(rows, seed, label_key):
    groups = build_split_components(rows, "video+duplicate")
    groups.sort(key=lambda group: hashlib.sha256(
        f"{seed}|{label_key}|{sorted(row.get('sample_id', '') for row in group)}".encode("utf-8")
    ).hexdigest())
    counts, policy = split_counts(len(groups))
    assignments = {}
    offset = 0
    for split_name in ("train", "validation", "test"):
        for group in groups[offset:offset + counts[split_name]]:
            for row in group:
                assignments[row["sample_id"]] = split_name
        offset += counts[split_name]
    return assignments, counts, policy


def resolve_group_mode(rows, requested):
    has_signer = any(row.get("signer_group_id", "").strip() or row.get("signer_id", "").strip() for row in rows)
    has_session = any(row.get("recording_session_id", "").strip() for row in rows)
    has_duplicate = any(row.get("duplicate_group_id", "").strip() for row in rows)
    if requested == "video":
        return "video+duplicate" if has_duplicate else "video"
    if requested == "signer":
        dimensions = [name for name, available in (("signer", has_signer), ("duplicate", has_duplicate)) if available]
        return "video+" + "+".join(dimensions) if dimensions else "video"
    if requested == "session":
        dimensions = [name for name, available in (("session", has_session), ("duplicate", has_duplicate)) if available]
        return "video+" + "+".join(dimensions) if dimensions else "video"
    dimensions = [
        name for name, available in (("signer", has_signer), ("session", has_session), ("duplicate", has_duplicate))
        if available
    ]
    return "video" if not dimensions else "video+" + "+".join(dimensions)


def row_group_tokens(row, mode):
    tokens = []
    source_id = row.get("source_id", "").strip()
    video_id = row.get("video_id", "").strip()
    if video_id:
        tokens.append(("video", source_id, video_id))
    if "signer" in mode:
        global_signer = row.get("signer_group_id", "").strip()
        signer_id = row.get("signer_id", "").strip()
        if global_signer:
            tokens.append(("global_signer", global_signer))
        elif signer_id:
            tokens.append(("source_signer", source_id, signer_id))
    if "session" in mode:
        session_id = row.get("recording_session_id", "").strip()
        if session_id:
            tokens.append(("source_session", source_id, session_id))
    if "duplicate" in mode:
        duplicate_group_id = row.get("duplicate_group_id", "").strip()
        if duplicate_group_id:
            tokens.append(("duplicate_group", duplicate_group_id))
    return tokens


def build_split_components(rows, mode):
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    first_by_token = {}
    for index, row in enumerate(rows):
        for token in row_group_tokens(row, mode):
            if token in first_by_token:
                union(index, first_by_token[token])
            else:
                first_by_token[token] = index

    components = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("video_id", "").strip():
            components[find(index)].append(row)
    return list(components.values())


def global_split_counts(group_count):
    if group_count <= 0:
        return {"train": 0, "validation": 0, "test": 0}, "no_video_reference"
    if group_count == 1:
        return {"train": 1, "validation": 0, "test": 0}, "train_only"
    if group_count == 2:
        return {"train": 1, "validation": 1, "test": 0}, "train_validation_only"
    validation_count = max(1, round(group_count * 0.15))
    test_count = max(1, round(group_count * 0.15))
    return {
        "train": group_count - validation_count - test_count,
        "validation": validation_count,
        "test": test_count,
    }, "train_validation_test"


def assign_global_splits(rows, seed, mode):
    components = build_split_components(rows, mode)
    components.sort(key=lambda group: hashlib.sha256(
        f"{seed}|{sorted((row.get('source_id', ''), row.get('video_id', '')) for row in group)}".encode("utf-8")
    ).hexdigest())
    counts, policy = global_split_counts(len(components))
    assignments = {}
    offset = 0
    for split_name in ("train", "validation", "test"):
        for group in components[offset:offset + counts[split_name]]:
            for row in group:
                assignments[row["sample_id"]] = split_name
        offset += counts[split_name]
    return assignments, counts, policy, len(components)


def actual_split_policy(counts):
    if counts["test"]:
        return "train_validation_test"
    if counts["validation"]:
        return "train_validation_only"
    if counts["train"]:
        return "train_only"
    return "no_video_reference"


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Plan VSL audit priorities and leakage-aware video splits.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("VSL_Dataset"))
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--group-by", choices=("auto", "video", "signer", "session"), default="auto")
    args = parser.parse_args()

    manifest_path = args.dataset_dir / "clean_vocabulary_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    by_gloss = defaultdict(list)
    video_glosses = defaultdict(set)
    for row in rows:
        label_key = row.get("canonical_gloss_id", "").strip() or row.get("provisional_gloss_id", "").strip()
        row["_label_key"] = label_key
        if row.get("video_id"):
            by_gloss[label_key].append(row)
            video_glosses[(row.get("source_id", ""), row["video_id"])].add(label_key)

    conflicting_videos = {key for key, glosses in video_glosses.items() if len(glosses) > 1}
    proposed_mode = resolve_group_mode(rows, args.group_by)
    proposal_rows = [
        row for row in rows
        if row.get("video_id") and (row.get("source_id", ""), row["video_id"]) not in conflicting_videos
    ]
    if proposed_mode == "video":
        proposed_assignments = {}
        proposed_group_count = len({(row.get("source_id", ""), row["video_id"]) for row in proposal_rows})
        for label_key, label_rows in by_gloss.items():
            usable = [row for row in label_rows if (row.get("source_id", ""), row["video_id"]) not in conflicting_videos]
            assignments, _, _ = assign_group_splits(usable, args.seed, label_key)
            proposed_assignments.update(assignments)
    else:
        proposed_assignments, _, _, proposed_group_count = assign_global_splits(proposal_rows, args.seed, proposed_mode)

    eligibility_reasons = Counter()
    eligible_rows = []
    for row in rows:
        row["final_split"] = ""
        video_key = (row.get("source_id", ""), row.get("video_id", ""))
        if not row.get("video_id"):
            row["proposed_split"] = "no_video_reference"
            row["final_eligibility_reason"] = "missing_video_reference"
            eligibility_reasons["missing_video_reference"] += 1
        elif video_key in conflicting_videos:
            row["proposed_split"] = "hold_conflicting_label"
            row["final_eligibility_reason"] = "video_referenced_with_multiple_glosses"
            eligibility_reasons["video_referenced_with_multiple_glosses"] += 1
        else:
            row["proposed_split"] = proposed_assignments.get(row["sample_id"], "hold_missing_split_group")
            eligible, reason = final_eligibility(row)
            if eligible:
                eligible_rows.append(row)
                row["final_eligibility_reason"] = "approved"
            else:
                row["final_eligibility_reason"] = reason
                eligibility_reasons[reason] += 1

    final_mode = resolve_group_mode(eligible_rows, args.group_by)
    if final_mode == "video":
        final_assignments = {}
        for label_key, label_rows in by_gloss.items():
            approved = [row for row in label_rows if row in eligible_rows]
            assignments, _, _ = assign_group_splits(approved, args.seed, label_key)
            final_assignments.update(assignments)
        final_group_count = len({(row.get("source_id", ""), row["video_id"]) for row in eligible_rows})
    else:
        final_assignments, _, _, final_group_count = assign_global_splits(eligible_rows, args.seed, final_mode)

    final_totals = Counter()
    for row in rows:
        row["final_split"] = final_assignments.get(row["sample_id"], "")
        row["split_unit"] = proposed_mode
        row["split_independence"] = (
            "signer_and_session_grouped" if "signer" in proposed_mode and "session" in proposed_mode
            else "signer_grouped" if "signer" in proposed_mode
            else "session_grouped_signer_unknown" if "session" in proposed_mode
            else "video_independent_signer_unknown"
        )
        if row.get("final_split"):
            final_totals[row["final_split"]] += 1

    proposed_totals = Counter(row["proposed_split"] for row in rows if row["proposed_split"] in ("train", "validation", "test"))
    priority_order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    gloss_rows = []
    coverage_bins = Counter()
    for label_key, label_rows in by_gloss.items():
        unique_video_keys = {(row.get("source_id", ""), row["video_id"]) for row in label_rows}
        target_counts, target_policy = split_counts(len(unique_video_keys))
        coverage_bins[target_policy] += 1
        proposal_counts = Counter(row["proposed_split"] for row in label_rows)
        approved = [row for row in label_rows if row in eligible_rows]
        final_counts = Counter(row["final_split"] for row in approved)
        final_policy = actual_split_policy(final_counts)
        mode_has_signer = "signer" in final_mode and all(
            row.get("signer_group_id", "").strip() or row.get("signer_id", "").strip() for row in approved
        )
        gloss_rows.append({
            "label_id": label_key,
            "gloss_candidate": label_rows[0].get("gloss_candidate", ""),
            "reference_videos": len(unique_video_keys),
            "audit_priority": audit_priority(len(unique_video_keys)),
            "reference_target_policy": target_policy,
            "target_train": target_counts["train"],
            "target_validation": target_counts["validation"],
            "target_test": target_counts["test"],
            "proposed_policy": actual_split_policy(proposal_counts),
            "proposed_train": proposal_counts["train"],
            "proposed_validation": proposal_counts["validation"],
            "proposed_test": proposal_counts["test"],
            "approved_training_videos": len({(row.get("source_id", ""), row.get("video_id")) for row in approved}),
            "final_policy": final_policy if approved else "no_approved_samples",
            "final_train": final_counts["train"],
            "final_validation": final_counts["validation"],
            "final_test": final_counts["test"],
            "signer_independent_split_possible": bool(mode_has_signer and final_counts["train"] and final_counts["test"]),
        })

    for row in rows:
        label_key = row.pop("_label_key", "")
        row["audit_priority"] = audit_priority(len(by_gloss.get(label_key, [])))
    rows.sort(key=lambda row: (priority_order[row["audit_priority"]], row.get("gloss_candidate", "").casefold(), row.get("video_id", "")))
    gloss_rows.sort(key=lambda row: (priority_order[row["audit_priority"]], row["gloss_candidate"].casefold()))

    row_fields = list(rows[0]) if rows else []
    write_csv(args.dataset_dir / "video_split_plan.csv", row_fields, rows)
    write_csv(
        args.dataset_dir / "gloss_split_plan.csv",
        [
            "label_id", "gloss_candidate", "reference_videos", "audit_priority", "reference_target_policy",
            "target_train", "target_validation", "target_test", "proposed_policy", "proposed_train",
            "proposed_validation", "proposed_test", "approved_training_videos", "final_policy",
            "final_train", "final_validation", "final_test", "signer_independent_split_possible",
        ],
        gloss_rows,
    )

    samples_by_split = lambda field: {name: Counter(row.get(field, "") for row in rows).get(name, 0) for name in ("train", "validation", "test")}
    signer_known = sum(bool(row.get("signer_id", "").strip() or row.get("signer_group_id", "").strip()) for row in rows)
    session_known = sum(bool(row.get("recording_session_id", "").strip()) for row in rows)
    summary = {
        "source_manifest": str(manifest_path),
        "split_seed": args.seed,
        "requested_group_by": args.group_by,
        "proposed_group_unit": proposed_mode,
        "proposed_group_count": proposed_group_count,
        "final_group_unit": final_mode,
        "final_group_count": final_group_count,
        "total_sample_rows": len(rows),
        "gloss_groups": len(gloss_rows),
        "gloss_groups_by_reference_split_policy": dict(sorted(coverage_bins.items())),
        "samples_by_proposed_split": samples_by_split("proposed_split"),
        "samples_blocked_for_cross_label_video_conflict": sum(1 for row in rows if row["proposed_split"] == "hold_conflicting_label"),
        "samples_by_final_split": samples_by_split("final_split"),
        "samples_blocked_by_audit_or_rights": dict(sorted(eligibility_reasons.items())),
        "signer_ids_known": signer_known,
        "recording_session_ids_known": session_known,
        "signer_independent_claim_supported": "signer" in final_mode and final_totals["test"] > 0,
        "note": "Proposed splits are audit-planning only. Final splits include only explicitly approved, accessible, correctly labelled, non-duplicate, rights-cleared samples. When IDs exist, signer and/or session groups are connected globally across glosses. Unknown identities fall back to video grouping and cannot support a signer-independent claim.",
    }
    (args.dataset_dir / "split_plan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()