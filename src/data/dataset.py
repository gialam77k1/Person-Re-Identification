from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from src.common.utils import resolve_path


class Market1501Dataset(Dataset):
    def __init__(self, folder: str | Path, transform=None, relabel: bool = False) -> None:
        self.folder = resolve_path(folder)
        self.transform = transform
        self.relabel = relabel
        self.samples: list[dict[str, int | str]] = []

        pid_container = set()
        for image_path in sorted(self.folder.glob("*.jpg")):
            pid = int(image_path.name.split("_")[0])
            if pid == -1:
                continue
            pid_container.add(pid)

        self.pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_container))}

        for image_path in sorted(self.folder.glob("*.jpg")):
            pid = int(image_path.name.split("_")[0])
            if pid == -1:
                continue
            camid = int(image_path.name.split("_")[1][1]) - 1
            mapped_pid = self.pid2label[pid] if relabel else pid
            self.samples.append(
                {
                    "img_path": str(image_path),
                    "pid": mapped_pid,
                    "camid": camid,
                }
            )

        self.num_classes = len(pid_container)
        self.labels = [int(sample["pid"]) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample["img_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "pid": int(sample["pid"]),
            "camid": int(sample["camid"]),
            "path": str(sample["img_path"]),
        }


class RandomIdentitySampler(Sampler[int]):
    def __init__(self, dataset: Market1501Dataset, batch_size: int, instances_per_identity: int) -> None:
        if batch_size % instances_per_identity != 0:
            raise ValueError("batch_size must be divisible by instances_per_identity")

        self.dataset = dataset
        self.batch_size = batch_size
        self.instances_per_identity = instances_per_identity
        self.identities_per_batch = batch_size // instances_per_identity

        self.index_dic: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(dataset.labels):
            self.index_dic[label].append(index)

        self.pids = list(self.index_dic.keys())
        self.length = self._compute_length()

    def _compute_length(self) -> int:
        total = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.instances_per_identity:
                num = self.instances_per_identity
            total += num - num % self.instances_per_identity
        return total

    def __iter__(self):
        batch_indices: list[int] = []
        pid_to_batches: dict[int, list[list[int]]] = {}

        for pid in self.pids:
            idxs = list(self.index_dic[pid])
            if len(idxs) < self.instances_per_identity:
                idxs = np_random_choice(idxs, self.instances_per_identity)
            random.shuffle(idxs)
            chunked = [
                idxs[i : i + self.instances_per_identity]
                for i in range(0, len(idxs), self.instances_per_identity)
                if len(idxs[i : i + self.instances_per_identity]) == self.instances_per_identity
            ]
            pid_to_batches[pid] = chunked

        available_pids = [pid for pid, chunks in pid_to_batches.items() if chunks]

        while len(available_pids) >= self.identities_per_batch:
            selected_pids = random.sample(available_pids, self.identities_per_batch)
            for pid in selected_pids:
                batch_indices.extend(pid_to_batches[pid].pop(0))
                if not pid_to_batches[pid]:
                    available_pids.remove(pid)

        return iter(batch_indices)

    def __len__(self) -> int:
        return self.length


def np_random_choice(items: list[int], size: int) -> list[int]:
    if not items:
        return []
    repeats = math.ceil(size / len(items))
    expanded = items * repeats
    random.shuffle(expanded)
    return expanded[:size]


def build_transforms(height: int, width: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, test_transform
