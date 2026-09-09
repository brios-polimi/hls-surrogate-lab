import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch_geometric  # noqa: F401
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "requires PyTorch and PyG")
class TrainingSmokeTests(unittest.TestCase):
    def test_empty_hierarchy_messages_keep_zero_gradient_paths(self):
        from ll_hls4ml.models.hierarchical import _messages, _reverse_messages

        projection = torch.nn.Linear(3, 4)
        edge_projection = torch.nn.Embedding(2, 4)
        empty_source = projection(torch.empty((0, 3)))
        empty_edge_features = edge_projection(torch.empty(0, dtype=torch.long))
        empty_edges = torch.empty((2, 0), dtype=torch.long)

        forward = _messages(
            empty_source,
            empty_edges,
            target_count=2,
            edge_features=empty_edge_features,
        )
        reverse = _reverse_messages(empty_source, empty_edges, source_count=2)
        (forward.sum() + reverse.sum()).backward()

        self.assertIsNotNone(projection.weight.grad)
        self.assertIsNotNone(projection.bias.grad)
        self.assertIsNotNone(edge_projection.weight.grad)
        self.assertTrue(torch.count_nonzero(projection.weight.grad) == 0)
        self.assertTrue(torch.count_nonzero(projection.bias.grad) == 0)
        self.assertTrue(torch.count_nonzero(edge_projection.weight.grad) == 0)

    def test_json_tensorization_and_training_for_both_models(self):
        from ll_hls4ml.data.dataset import HeteroGraphDataset
        from ll_hls4ml.data.tensorize import create_graph_tensors
        from ll_hls4ml.models.registry import build
        from ll_hls4ml.training.loaders import make_loader
        from ll_hls4ml.training.loops import fit, validate_one_epoch
        from ll_hls4ml.training.targets import compute_target_z_stats
        from ll_hls4ml.training.targets import LogHuberHurdleLoss

        graph = {
            "nodes": [
                {"id": 0, "type": 0, "text": "add", "function": 0, "block": 0},
                {"id": 1, "type": 1, "text": "ap_fixed<16, 6, AP_TRN, AP_WRAP>"},
                {"id": 2, "type": 2, "text": "i32"},
                {
                    "id": 3,
                    "type": 3,
                    "text": "pragma.pipeline",
                    "features": {
                        "schema_version": ["2"],
                        "arguments_json": [
                            json.dumps({"ii": ["2"], "rewind": ["true"]})
                        ],
                    },
                },
                {
                    "id": 4,
                    "type": 4,
                    "text": "llvm.basic_block",
                    "function": 0,
                    "block": 0,
                    "features": {
                        "name": ["ReuseLoop"],
                        "is_source_loop": ["true"],
                    },
                },
                {
                    "id": 5,
                    "type": 5,
                    "text": "llvm.function",
                    "function": 0,
                    "block": -1,
                    "features": {"name": ["kernel"], "is_defined": ["true"]},
                },
            ],
            "links": [
                {"source": 0, "target": 0, "relation": "control", "position": 0},
                {"source": 0, "target": 1, "relation": "defines", "position": 0},
                {"source": 1, "target": 0, "relation": "operand", "position": 1},
                {"source": 2, "target": 0, "relation": "operand", "position": 2},
                {"source": 3, "target": 0, "relation": "applies_to", "position": 0},
                {"source": 3, "target": 1, "relation": "applies_to", "position": 0},
                {"source": 3, "target": 4, "relation": "applies_to", "position": 0},
            ],
        }
        labels = [100, 200, 3, 4, 1000, 10]
        instruction_vocab = {"UNK": 0, "add": 1}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            graph_dir = root / "graphs"
            archive_dir = graph_dir / "smoke" / "archive_1"
            exemplar_dir = graph_dir / "exemplar" / "archive_1"
            archive_dir.mkdir(parents=True)
            exemplar_dir.mkdir(parents=True)
            for index in range(8):
                sample = dict(graph)
                sample["labels"] = {
                    key: value + index
                    for key, value in zip(
                        (
                            "lut",
                            "ff",
                            "dsp",
                            "bram",
                            "cycles_max",
                            "interval_max",
                        ),
                        labels,
                    )
                }
                (archive_dir / f"sample_{index}.json").write_text(json.dumps(sample))
                if index < 2:
                    (exemplar_dir / f"sample_{index}.json").write_text(
                        json.dumps(sample)
                    )

            tensor_dir = root / "tensors"
            create_graph_tensors(
                graph_dir,
                tensor_dir,
                instruction_vocab,
                kernel_subset=["smoke", "exemplar"],
                n_workers=1,
            )
            dataset = HeteroGraphDataset(tensor_dir, types=["smoke"])
            self.assertTrue((tensor_dir / "labels.json").exists())
            self.assertEqual(dataset.targets.shape, (8, 6))
            y_means, y_stds = compute_target_z_stats(dataset)
            loader = make_loader(
                dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Smoke training device: {device}")

            for name, extra in (
                ("mlp", {}),
                (
                    "hetero_gat",
                    {
                        "edge_pos_vocab_size": 2,
                        "pool": "multi",
                        "use_global_features": True,
                        "use_context": True,
                        "split_heads": True,
                        "hurdle_heads": True,
                    },
                ),
                (
                    "hierarchical",
                    {
                        "edge_pos_vocab_size": 2,
                        "use_context": True,
                        "split_heads": True,
                        "hurdle_heads": True,
                    },
                ),
                (
                    "hierarchical_no_local_message",
                    {
                        "edge_pos_vocab_size": 2,
                        "use_context": True,
                        "split_heads": True,
                        "hurdle_heads": True,
                    },
                ),
                *(
                    (
                        name,
                        {
                            "edge_pos_vocab_size": 2,
                            "use_context": True,
                            "split_heads": True,
                            "hurdle_heads": True,
                            "attention_heads": 4,
                            "attention_layers": 1,
                            "cfg_recurrent_steps": 2,
                        },
                    )
                    for name in (
                        "hierarchical_sequence",
                        "hierarchical_block_attention",
                        "hierarchical_memory_dual",
                    )
                ),
            ):
                model = build(
                    name,
                    instruction_vocab_size=len(instruction_vocab),
                    y_means=y_means,
                    y_stds=y_stds,
                    hidden_dim=16,
                    num_layers=1,
                    dropout=0.0,
                    **extra,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = (
                    LogHuberHurdleLoss(y_means, y_stds)
                    if extra.get("hurdle_heads")
                    else torch.nn.HuberLoss()
                )
                model = fit(
                    model,
                    loader,
                    loader,
                    epochs=1,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=None,
                    device=device,
                    patience=0,
                    verbose=0,
                    experiment_name=f"smoke_{name}",
                    checkpoint_dir=root / "checkpoints",
                )
                metrics = validate_one_epoch(
                    model,
                    loader,
                    criterion,
                    device,
                )
                self.assertTrue(torch.isfinite(metrics["r2"]).all())
                self.assertTrue((root / "checkpoints" / f"smoke_{name}_checkpoint.pt").is_file())

            vocab_path = root / "vocab.json"
            vocab_path.write_text(
                json.dumps({"vocab": instruction_vocab, "max_pos": 2})
            )
            config_path = root / "train.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "cli_smoke",
                        "model": "mlp",
                        "tensor_dir": str(tensor_dir),
                        "vocab_path": str(vocab_path),
                        "checkpoint_dir": str(root / "cli_checkpoints"),
                        "results_dir": str(root / "results"),
                        "kernel_types": ["smoke"],
                        "seed": 42,
                        "val_fraction": 0.125,
                        "test_fraction": 0.125,
                        "train_scale": 1.0,
                        "baseline_archives_per_family": 1,
                        "evaluation_archives_per_family": 1,
                        "batch_size": 4,
                        "num_workers": 0,
                        "epochs": 1,
                        "patience": 2,
                        "hidden_dim": 16,
                        "num_layers": 1,
                        "dropout": 0.0,
                    }
                )
            )
            repository = Path(__file__).resolve().parents[1]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repository / "src")
            environment["LL_HLS4ML_TQDM"] = "0"
            subprocess.run(
                [
                    sys.executable,
                    str(repository / "scripts" / "train.py"),
                    "--config",
                    str(config_path),
                ],
                check=True,
                cwd=repository,
                env=environment,
            )
            result_dir = root / "results" / "cli_smoke"
            result = json.loads((result_dir / "summary.json").read_text())
            self.assertEqual(
                result["sizes"],
                {"train": 6, "validation": 1, "test": 1, "exemplar": 2},
            )
            self.assertEqual(
                {
                    row["kernel_family"]
                    for row in result["metrics"]
                },
                {"all", "smoke", "exemplar"},
            )
            self.assertTrue((result_dir / "metrics.csv").is_file())
            self.assertTrue((result_dir / "predictions.csv").is_file())
            self.assertTrue((result_dir / "learning_curves.csv").is_file())
            self.assertTrue((result_dir / "macro_metrics.csv").is_file())
            self.assertTrue(
                (result_dir / "structural_error_slices.csv").is_file()
            )
            self.assertTrue((result_dir / "split_manifest.json").is_file())
            self.assertTrue(
                (result_dir / "data_scale_manifest.json").is_file()
            )
            self.assertTrue((result_dir / "figures" / "test__rpe.png").is_file())
            self.assertTrue((result_dir / "figures" / "test__scatter.png").is_file())
            subprocess.run(
                [
                    sys.executable,
                    str(repository / "scripts" / "train.py"),
                    "--config",
                    str(config_path),
                    "--evaluate-checkpoint",
                    str(
                        root
                        / "cli_checkpoints"
                        / "cli_smoke_checkpoint.pt"
                    ),
                ],
                check=True,
                cwd=repository,
                env=environment,
            )
            evaluated_result = json.loads(
                (result_dir / "summary.json").read_text()
            )
            self.assertEqual(evaluated_result["best_epoch"], 1)
            self.assertEqual(
                [row["epoch"] for row in evaluated_result["training_history"]],
                [1],
            )

            if os.environ.get("LL_HLS4ML_SKIP_DDP_SMOKE") == "1":
                return

            ddp_environment = environment.copy()
            ddp_environment["CUDA_VISIBLE_DEVICES"] = ""
            for model_name in (
                "hierarchical_sequence",
                "hierarchical_block_attention",
                "hierarchical_memory_dual",
            ):
                experiment = f"cli_ddp_smoke_{model_name}"
                ddp_config = json.loads(config_path.read_text())
                ddp_config.update(
                    {
                        "experiment_name": experiment,
                        "model": model_name,
                        "batch_size": 2,
                        "epochs": 1,
                        "use_context": True,
                        "split_heads": True,
                        "hurdle_heads": True,
                        "loss": "log_huber_hurdle",
                        "attention_heads": 4,
                        "attention_layers": 1,
                        "cfg_recurrent_steps": 2,
                    }
                )
                config_path.write_text(json.dumps(ddp_config))
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=2",
                        str(repository / "scripts" / "train.py"),
                        "--config",
                        str(config_path),
                    ],
                    check=True,
                    cwd=repository,
                    env=ddp_environment,
                    timeout=60,
                )
                self.assertTrue(
                    (root / "results" / experiment / "summary.json").is_file()
                )


if __name__ == "__main__":
    unittest.main()
