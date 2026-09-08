"""PyG HeteroData dataset over preprocessed .pt files."""
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from ll_hls4ml.io.schema import LABEL_KEYS


class HeteroGraphDataset(Dataset):
    """
    Lazy-loading dataset for HeteroData .pt graphs.
    Indexes the filesystem at init time, loads graphs on demand.
    """

    def __init__(
        self,
        root: str | Path,
        types: list[str] | None = None,
        max_per_type: dict[str, int] | int | None = None,
        transform=None,
        silent: bool = True,
        deduplicate_graph_ids: bool = True,
        relative_paths: list[str] | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        if relative_paths is None:
            self.paths, self.duplicate_paths = self._index(
                types,
                max_per_type,
                silent,
                deduplicate_graph_ids,
            )
        else:
            self.paths = [self.root / path for path in relative_paths]
            self.duplicate_paths = []
        self.targets, self.metadata = self._load_index()

    def _load_index(self) -> tuple[torch.Tensor | None, list[dict]]:
        """Load labels and split/context metadata without deserializing tensors."""
        index_path = self.root / "labels.json"
        if not index_path.exists():
            return None, [{} for _ in self.paths]
        try:
            with index_path.open() as handle:
                index = json.load(handle)
            if index["label_keys"] != LABEL_KEYS:
                return None, [{} for _ in self.paths]
            labels = index["labels"]
            metadata = index.get("metadata", {})
            relative_paths = [
                path.relative_to(self.root).as_posix() for path in self.paths
            ]
            values = [labels[path] for path in relative_paths]
            metadata_values = [metadata.get(path, {}) for path in relative_paths]
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return None, [{} for _ in self.paths]
        return torch.tensor(values, dtype=torch.float), metadata_values

    def _index(
        self,
        types: list[str] | None,
        max_per_type: dict[str, int] | int | None,
        silent: bool,
        deduplicate_graph_ids: bool,
    ) -> tuple[list[Path], list[Path]]:
        paths = []
        duplicate_paths = []
        graph_ids = set()
        root = self.root

        type_dirs = (
            [root / t for t in types]
            if types
            else [p for p in root.iterdir() if p.is_dir()]
        )

        type_counts = defaultdict(int)
        type_sizes = defaultdict(int)
        for type_dir in sorted(type_dirs):
            if not type_dir.exists():
                raise FileNotFoundError(f"Type directory not found: {type_dir}")
            for archive_dir in sorted(type_dir.iterdir()):
                if not archive_dir.is_dir():
                    continue

                if isinstance(max_per_type, dict):
                    max_this_kernel = max_per_type.get(type_dir.name, None)
                elif isinstance(max_per_type, int):
                    max_this_kernel = max_per_type
                else:
                    max_this_kernel = None

                for pt_file in sorted(archive_dir.glob("*.pt")):
                    if max_this_kernel is not None and type_counts[type_dir.name] >= max_this_kernel:
                        break
                    if deduplicate_graph_ids and pt_file.stem in graph_ids:
                        duplicate_paths.append(pt_file)
                        continue
                    graph_ids.add(pt_file.stem)
                    type_counts[type_dir.name] += 1
                    paths.append(pt_file)
                    type_sizes[type_dir.name] += pt_file.stat().st_size

        if not silent:
            print(f"Indexed {len(paths)} graphs across {len(type_dirs)} type(s)")
            if duplicate_paths:
                print(
                    f"  skipped {len(duplicate_paths)} duplicate graph UUID path(s)"
                )
            for type_name, count in type_counts.items():
                print(f"  {type_name}: {count}")
                print(f"    avg size: {type_sizes[type_name] / count / 1024 / 1024 :.2f} MB")
                print(f"    total size: {type_sizes[type_name] / 1024 / 1024 :.2f} MB")
        return paths, duplicate_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> HeteroData:
        data = torch.load(self.paths[idx], weights_only=False)
        data.graph_id = self.paths[idx].stem
        if self.transform:
            data = self.transform(data)
        return data

    def type_of(self, idx: int) -> str:
        """Return kernel type (e.g. 'exemplar') for a given index."""
        return self.paths[idx].parts[-3]

    def metadata_of(self, idx: int) -> dict:
        return self.metadata[idx]
