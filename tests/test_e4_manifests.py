import sys
from pathlib import Path
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from build_e4_manifests import build_folds
from ll_hls4ml.data.signatures import (
    HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION,
    HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
)


class E4ManifestTests(unittest.TestCase):
    @staticmethod
    def source(version=HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION):
        manifest = {name: [] for name in ("train", "validation", "test", "exemplar")}
        for family in ("dense", "conv"):
            for group in range(5):
                for sample in range(2):
                    split = ("train", "validation", "test")[group % 3]
                    manifest[split].append({
                        "kernel_family": family,
                        "architecture_id": f"{family}-{group}",
                        "architecture_signature_version": version,
                        "topology_signature_version": HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
                        "tensor_path": f"{family}/archive_1/{group}-{sample}.pt",
                        "original_dataset_split": split,
                    })
        manifest["exemplar"].append({
            "kernel_family": "exemplar",
            "architecture_id": "exemplar-0",
            "architecture_signature_version": version,
            "topology_signature_version": HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
            "tensor_path": "exemplar/archive_1/x.pt",
            "original_dataset_split": "exemplar",
        })
        return manifest

    def test_folds_hold_out_exactly_one_family_and_preserve_universe(self):
        source = self.source()
        folds, audit = build_folds(source, seed=42)
        expected = {
            row["tensor_path"] for rows in source.values() for row in rows
        }
        for family, manifest in folds.items():
            self.assertEqual(
                {row["kernel_family"] for row in manifest["test"]}, {family}
            )
            self.assertEqual(
                {
                    row["tensor_path"]
                    for rows in manifest.values() for row in rows
                },
                expected,
            )
            self.assertEqual(audit["folds"][family]["held_out_family"], family)

    def test_stale_coarse_signature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact ordered-layer"):
            build_folds(self.source("wa_hls4ml_family_layer_count_v1"))


if __name__ == "__main__":
    unittest.main()
