"""Manifest-only experiment protocols for frozen learning releases."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib


SPLIT_NAMES = ("train", "validation", "test", "exemplar")


def _rank(seed: int, *parts: str) -> str:
    return hashlib.sha256("\0".join((str(seed), *parts)).encode()).hexdigest()


def _manifest(rows_by_split: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {
        name: sorted(rows_by_split.get(name, []), key=lambda row: row["tensor_path"])
        for name in SPLIT_NAMES
    }


def official_protocol(samples: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        split = "validation" if row["original_dataset_split"] in {"val", "validation"} else row["original_dataset_split"]
        if row["kernel_family"] == "exemplar":
            split = "exemplar"
        if split not in SPLIT_NAMES:
            raise ValueError(f"Invalid original split {split!r} for {row['tensor_path']}")
        result[split].append(row)
    return _manifest(result)


def architecture_grouped_protocol(
    samples: list[dict], *, seed: int = 42, val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, list[dict]]:
    """Assign whole architecture groups within family.

    The greedy objective balances sample counts and DSP/BRAM presence only.
    It never consumes target magnitudes.
    """
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Grouped fractions must be positive and sum to less than one")
    result: dict[str, list[dict]] = defaultdict(list)
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in samples:
        if row["kernel_family"] == "exemplar":
            result["exemplar"].append(row)
        else:
            grouped[row["kernel_family"]][row["architecture_id"]].append(row)

    for family, groups in sorted(grouped.items()):
        if len(groups) < 3:
            # Fixed-topology families cannot contribute an honest held-out
            # within-family signature. Retain them as training context only.
            result["train"].extend(row for rows in groups.values() for row in rows)
            continue
        fractions = {
            "train": 1.0 - val_fraction - test_fraction,
            "validation": val_fraction,
            "test": test_fraction,
        }

        def group_vector(rows):
            vector = [float(len(rows)), 0.0, 0.0]
            for row in rows:
                labels = row.get("labels")
                validity = row.get("label_validity_mask")
                for position, target_index in enumerate((2, 3), start=1):
                    valid = (
                        labels is not None and len(labels) > target_index
                        and labels[target_index] is not None
                        and (validity is None or validity[target_index])
                    )
                    vector[position] += float(valid and float(labels[target_index]) > 0)
            return tuple(vector)

        vectors = {group: group_vector(rows) for group, rows in groups.items()}
        totals = tuple(sum(vector[index] for vector in vectors.values()) for index in range(3))
        targets = {
            split: tuple(fraction * value for value in totals)
            for split, fraction in fractions.items()
        }
        assigned = {split: [0.0, 0.0, 0.0] for split in fractions}
        ordered = sorted(
            groups,
            key=lambda group: (-len(groups[group]), _rank(seed, family, group)),
        )

        def objective(destination, vector):
            score = 0.0
            for split in fractions:
                for index in range(3):
                    value = assigned[split][index]
                    if split == destination:
                        value += vector[index]
                    scale = max(1.0, targets[split][index])
                    score += ((value - targets[split][index]) / scale) ** 2
            return score

        # Guarantee every sufficiently diverse family contributes to all splits.
        for destination, group in zip(("train", "validation", "test"), ordered[:3]):
            vector = vectors[group]
            result[destination].extend(groups[group])
            assigned[destination] = [
                value + addition
                for value, addition in zip(assigned[destination], vector, strict=True)
            ]
        for group in ordered[3:]:
            vector = vectors[group]
            destination = min(
                fractions,
                key=lambda split: (
                    objective(split, vector), _rank(seed, family, group, split)
                ),
            )
            result[destination].extend(groups[group])
            assigned[destination] = [
                value + addition
                for value, addition in zip(assigned[destination], vector, strict=True)
            ]
    manifest = _manifest(result)
    if not manifest["validation"] or not manifest["test"]:
        raise ValueError("No family has enough architecture groups for grouped evaluation")
    return manifest


def leave_one_family_out_protocols(
    samples: list[dict], *, seed: int = 42, val_fraction: float = 0.15,
) -> dict[str, dict[str, list[dict]]]:
    """Build one fold per synthetic family; exemplars remain evaluation-only."""
    main = [row for row in samples if row["kernel_family"] != "exemplar"]
    exemplar = [row for row in samples if row["kernel_family"] == "exemplar"]
    families = sorted({row["kernel_family"] for row in main})
    folds = {}
    for held_out in families:
        result: dict[str, list[dict]] = defaultdict(list)
        result["test"] = [row for row in main if row["kernel_family"] == held_out]
        result["exemplar"] = exemplar
        training_pool = [row for row in main if row["kernel_family"] != held_out]
        by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in training_pool:
            by_group[(row["kernel_family"], row["architecture_id"])].append(row)
        by_family: dict[str, list[tuple[tuple[str, str], list[dict]]]] = defaultdict(list)
        for key, rows in by_group.items():
            by_family[key[0]].append((key, rows))
        for family, groups in by_family.items():
            if len(groups) < 2:
                # A coarse family fallback cannot be used to form validation
                # groups. Preserve the benchmark's pre-existing validation
                # membership and use the remaining source-family samples for fit.
                family_rows = [row for _key, rows in groups for row in rows]
                validation = [
                    row for row in family_rows
                    if row["original_dataset_split"] in {"val", "validation"}
                ]
                if not validation:
                    raise ValueError(
                        f"{family} has one architecture group and no official validation rows"
                    )
                result["validation"].extend(validation)
                result["train"].extend(row for row in family_rows if row not in validation)
                continue
            ordered = sorted(groups, key=lambda item: _rank(seed, held_out, *item[0]))
            target = max(1, round(sum(len(rows) for _, rows in ordered) * val_fraction))
            count = 0
            for _key, rows in ordered:
                destination = "validation" if count < target else "train"
                result[destination].extend(rows)
                if destination == "validation":
                    count += len(rows)
        folds[held_out] = _manifest(result)
    return folds


def audit_protocol(manifest: dict[str, list[dict]], *, group_key: str | None = None) -> dict:
    paths = [row["tensor_path"] for rows in manifest.values() for row in rows]
    duplicates = [path for path, count in Counter(paths).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate manifest paths: {duplicates[:5]}")
    if group_key:
        destinations: dict[tuple[str, str], set[str]] = defaultdict(set)
        for split in ("train", "validation", "test"):
            for row in manifest[split]:
                destinations[(row["kernel_family"], row[group_key])].add(split)
        leaks = [key for key, values in destinations.items() if len(values) > 1]
        if leaks:
            raise ValueError(f"{group_key} crosses splits: {leaks[:5]}")
    report = {
        "sizes": {name: len(manifest[name]) for name in SPLIT_NAMES},
        "families": {
            name: dict(Counter(row["kernel_family"] for row in manifest[name]))
            for name in SPLIT_NAMES
        },
        "architecture_groups": {
            name: len({row["architecture_id"] for row in manifest[name]})
            for name in SPLIT_NAMES
        },
    }
    if group_key:
        group_sizes = {
            name: Counter(
                (row["kernel_family"], row[group_key]) for row in manifest[name]
            )
            for name in ("train", "validation", "test")
        }
        report["group_counts_by_family"] = {
            name: dict(Counter(family for family, _group in sizes))
            for name, sizes in group_sizes.items()
        }
        report["maximum_group_size"] = {
            name: max(sizes.values(), default=0)
            for name, sizes in group_sizes.items()
        }
        report["target_presence_rates"] = {}
        for name in ("train", "validation", "test"):
            rows = manifest[name]
            report["target_presence_rates"][name] = {
                target: (
                    sum(
                        row.get("labels") is not None
                        and row["labels"][index] is not None
                        and float(row["labels"][index]) > 0
                        for row in rows
                    ) / len(rows)
                    if rows else None
                )
                for target, index in (("dsp", 2), ("bram", 3))
            }
    return report
