import unittest

from ll_hls4ml.data.protocols import (
    architecture_grouped_protocol,
    audit_protocol,
    leave_one_family_out_protocols,
)
from ll_hls4ml.data.signatures import (
    canonical_model_name,
    high_level_signature_fields,
    signature_fields,
)


class SignatureTests(unittest.TestCase):
    def test_synthesis_knobs_do_not_change_architecture(self):
        left = canonical_model_name("dense_104_128_104_10b", "2layer")
        right = canonical_model_name("dense_104_128_104_12b", "2layer")
        self.assertEqual(left, right)
        self.assertEqual(
            canonical_model_name(
                "model_Dense_16in_8out_ap_fixed<8, 4>_16rf_L", "rule4ml"
            ),
            "rule4ml:model_Dense_16in_8out",
        )

    def test_missing_model_name_is_explicit_family_fallback(self):
        fields = signature_fields("conv1d")
        self.assertEqual(fields["architecture_source"], "family_fallback")
        self.assertEqual(fields["architecture_summary"], "conv1d")

    def test_high_level_architecture_excludes_synthesis_knobs(self):
        columns = (
            "d_in1", "d_in2", "d_in3", "d_out1", "d_out2", "d_out3",
            "prec", "rf", "strategy", "layer_type", "activation_type",
            "filters", "kernel_size", "stride", "padding", "pooling",
            "batchnorm", "io_type",
        )
        left = [[8, 0, 0, 16, 0, 0, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
        right = [[8, 0, 0, 16, 0, 0, 16, 64, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]]
        self.assertEqual(
            high_level_signature_fields("dense", left, columns)["architecture_id"],
            high_level_signature_fields("dense", right, columns)["architecture_id"],
        )

    def test_high_level_architecture_retains_order_and_dimensions(self):
        columns = (
            "d_in1", "d_in2", "d_in3", "d_out1", "d_out2", "d_out3",
            "prec", "rf", "strategy", "layer_type", "activation_type",
            "filters", "kernel_size", "stride", "padding", "pooling",
            "batchnorm", "io_type",
        )
        first = [8, 0, 0, 16, 0, 0, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        second = [16, 0, 0, 4, 0, 0, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        base = high_level_signature_fields("dense", [first, second], columns)
        reordered = high_level_signature_fields("dense", [second, first], columns)
        resized = [row[:] for row in (first, second)]
        resized[0][3] = 32
        changed = high_level_signature_fields("dense", resized, columns)
        self.assertNotEqual(base["architecture_id"], reordered["architecture_id"])
        self.assertNotEqual(base["architecture_id"], changed["architecture_id"])


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def samples():
        rows = []
        for family in ("dense", "conv"):
            for group in range(5):
                for sample in range(2):
                    rows.append({
                        "kernel_family": family,
                        "architecture_id": f"{family}-{group}",
                        "tensor_path": f"{family}/archive_1/{group}-{sample}.pt",
                        "original_dataset_split": "train",
                    })
        rows.append({
            "kernel_family": "exemplar", "architecture_id": "exemplar-0",
            "tensor_path": "exemplar/archive_1/x.pt",
            "original_dataset_split": "exemplar",
        })
        return rows

    def test_architecture_groups_are_disjoint(self):
        manifest = architecture_grouped_protocol(self.samples(), seed=7)
        report = audit_protocol(manifest, group_key="architecture_id")
        self.assertGreater(report["sizes"]["train"], 0)
        self.assertGreater(report["sizes"]["validation"], 0)
        self.assertGreater(report["sizes"]["test"], 0)

    def test_grouped_assignment_balances_presence_without_splitting_groups(self):
        rows = []
        for group in range(30):
            for sample in range(2):
                rows.append({
                    "kernel_family": "dense",
                    "architecture_id": f"dense-{group}",
                    "tensor_path": f"dense/archive_1/{group}-{sample}.pt",
                    "original_dataset_split": "train",
                    "labels": [1, 1, float(group % 2), float(group % 3 == 0), 1, 1],
                    "label_validity_mask": [True] * 6,
                })
        manifest = architecture_grouped_protocol(rows, seed=42)
        report = audit_protocol(manifest, group_key="architecture_id")
        for split, expected in (("train", 42), ("validation", 9), ("test", 9)):
            self.assertLessEqual(abs(report["sizes"][split] - expected), 2)
        for target in ("dsp", "bram"):
            rates = [
                report["target_presence_rates"][split][target]
                for split in ("train", "validation", "test")
            ]
            self.assertLess(max(rates) - min(rates), 0.2)

    def test_family_fallback_is_training_only_in_grouped_split(self):
        rows = []
        for family in ("dense", "conv"):
            architecture = signature_fields(family)["architecture_id"]
            rows.append({
                "kernel_family": family, "architecture_id": architecture,
                "tensor_path": f"{family}/archive_1/x.pt",
                "original_dataset_split": "train",
            })
        with self.assertRaisesRegex(ValueError, "No family"):
            architecture_grouped_protocol(rows)

    def test_leave_family_out_has_exact_test_family(self):
        folds = leave_one_family_out_protocols(self.samples())
        for family, manifest in folds.items():
            self.assertEqual(
                {row["kernel_family"] for row in manifest["test"]}, {family}
            )

    def test_leave_family_out_family_fallback_uses_official_validation(self):
        rows = []
        for family in ("dense", "conv"):
            for index, split in enumerate(("train", "validation", "test")):
                rows.append({
                    "kernel_family": family, "architecture_id": family,
                    "tensor_path": f"{family}/archive_1/{index}.pt",
                    "original_dataset_split": split,
                })
        fold = leave_one_family_out_protocols(rows)["dense"]
        self.assertEqual(len(fold["validation"]), 1)
        self.assertEqual(len(fold["train"]), 2)


if __name__ == "__main__":
    unittest.main()
