from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from src.common.utils import resolve_path


class ImageFolderReIDDataset(Dataset):
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


class Market1501Dataset(ImageFolderReIDDataset):
    pass


DATASET_REGISTRY = {
    "market1501": ImageFolderReIDDataset,
    "market-1501": ImageFolderReIDDataset,
    "dukemtmc": ImageFolderReIDDataset,
    "dukemtmc-reid": ImageFolderReIDDataset,
    "msmt17": ImageFolderReIDDataset,
}


def get_dataset_class(dataset_name: str):
    dataset_key = dataset_name.strip().lower()
    if dataset_key not in DATASET_REGISTRY:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")
    return DATASET_REGISTRY[dataset_key]


def build_dataset(
    config: dict,
    split: str,
    transform=None,
    relabel: bool = False,
):
    source_type = config["data"].get("source_type", "image_folder")
    if source_type != "image_folder":
        raise ValueError(
            f"Unsupported data.source_type '{source_type}' for dataset loading. "
            "Currently supported: image_folder"
        )

    dataset_name = config["data"]["dataset"]["name"]
    dataset_class = get_dataset_class(dataset_name)
    split_to_dir = {
        "train": config["data"]["train_dir"],
        "query": config["data"]["query_dir"],
        "gallery": config["data"]["gallery_dir"],
    }
    if split not in split_to_dir:
        raise ValueError(f"Unsupported split '{split}'. Expected one of: train, query, gallery")
    return dataset_class(split_to_dir[split], transform=transform, relabel=relabel)


def build_dataset_splits(
    config: dict,
    train_transform=None,
    test_transform=None,
):
    train_dataset = build_dataset(config, "train", transform=train_transform, relabel=True)
    query_dataset = build_dataset(config, "query", transform=test_transform, relabel=False)
    gallery_dataset = build_dataset(config, "gallery", transform=test_transform, relabel=False)
    return train_dataset, query_dataset, gallery_dataset


class RandomIdentitySampler(Sampler[int]):
    def __init__(self, dataset: ImageFolderReIDDataset, batch_size: int, instances_per_identity: int) -> None:
        if batch_size % instances_per_identity != 0:
            raise ValueError("batch_size must be divisible by instances_per_identity")

        self.dataset = dataset
        self.batch_size = batch_size
        self.instances_per_identity = instances_per_identity
        self.identities_per_batch = batch_size // instances_per_identity

        self.index_dic: dict[int, list[int]] = defaultdict(list)
        self.index_cam_dic: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, label in enumerate(dataset.labels):
            self.index_dic[label].append(index)
            camid = int(dataset.samples[index]["camid"])
            self.index_cam_dic[label][camid].append(index)

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
            idxs = self._sample_pid_indices(pid)
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

    def _sample_pid_indices(self, pid: int) -> list[int]:
        idxs = list(self.index_dic[pid])
        if len(idxs) < self.instances_per_identity:
            return np_random_choice(idxs, self.instances_per_identity)

        camera_to_indices = {camid: list(indices) for camid, indices in self.index_cam_dic[pid].items()}
        for indices in camera_to_indices.values():
            random.shuffle(indices)

        sampled_indices: list[int] = []
        while True:
            available_cams = [camid for camid, indices in camera_to_indices.items() if indices]
            if not available_cams:
                break

            random.shuffle(available_cams)
            group: list[int] = []
            for camid in available_cams:
                if len(group) >= self.instances_per_identity:
                    break
                group.append(camera_to_indices[camid].pop())

            if len(group) < self.instances_per_identity:
                remaining = [index for indices in camera_to_indices.values() for index in indices]
                while len(group) < self.instances_per_identity and remaining:
                    random.shuffle(remaining)
                    picked = remaining.pop()
                    group.append(picked)
                    for indices in camera_to_indices.values():
                        if picked in indices:
                            indices.remove(picked)
                            break

            if len(group) == self.instances_per_identity:
                sampled_indices.extend(group)
            else:
                break

        return sampled_indices if sampled_indices else np_random_choice(idxs, self.instances_per_identity)


def np_random_choice(items: list[int], size: int) -> list[int]:
    if not items:
        return []
    repeats = math.ceil(size / len(items))
    expanded = items * repeats
    random.shuffle(expanded)
    return expanded[:size]


def build_transforms(
    height: int,
    width: int,
    color_jitter: bool = False,
    random_erasing: bool = False,
    random_grayscale_p: float = 0.0,
    random_affine_degrees: float = 0.0,
    random_occlusion_p: float = 0.0,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transforms = [
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((height, width)),
    ]
    if color_jitter:
        train_transforms.append(
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05)
        )
    if random_affine_degrees > 0.0:
        train_transforms.append(
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=random_affine_degrees,
                        translate=(0.03, 0.03),
                        scale=(0.95, 1.05),
                        shear=5,
                    )
                ],
                p=0.4,
            )
        )
    if random_grayscale_p > 0.0:
        train_transforms.append(transforms.RandomGrayscale(p=random_grayscale_p))

    train_transforms.extend(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    if random_erasing:
        train_transforms.append(
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    if random_occlusion_p > 0.0:
        train_transforms.append(
            transforms.RandomErasing(
                p=random_occlusion_p,
                scale=(0.12, 0.28),
                ratio=(0.8, 1.8),
                value="random",
            )
        )

    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, test_transform
