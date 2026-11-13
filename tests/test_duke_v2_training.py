from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
import torch

from src.models.transfer import is_task_head_parameter, set_vit_lower_blocks_trainable
from src.train import collect_gpu_telemetry
from src.validate_duke_v2_training import MINIMUM_FREE_BYTES, validate_inputs


ROOT = Path(__file__).resolve().parents[1]


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class FakeModule:
    def __init__(self, count: int = 1) -> None:
        self._parameters = [FakeParameter() for _ in range(count)]

    def parameters(self):
        return iter(self._parameters)


class FakeEncoder(FakeModule):
    def __init__(self) -> None:
        super().__init__(0)
        self.layers = [FakeModule() for _ in range(12)]
        self.pos_embedding = FakeParameter()


class FakeBackbone(FakeModule):
    def __init__(self) -> None:
        super().__init__(0)
        self.conv_proj = FakeModule()
        self.class_token = FakeParameter()
        self.encoder = FakeEncoder()

    def parameters(self):
        yield from self.conv_proj.parameters()
        yield self.class_token
        yield self.encoder.pos_embedding
        for layer in self.encoder.layers:
            yield from layer.parameters()


class FakeModel:
    def __init__(self) -> None:
        self.backbone = FakeBackbone()


class DukeV2TrainingTest(unittest.TestCase):
    def test_cpu_training_telemetry_does_not_require_nvidia_smi(self) -> None:
        self.assertEqual(collect_gpu_telemetry(torch.device("cpu")), {})

    def test_config_matches_approved_training_contract(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/duke_v2_train.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["release"]["id"], "market1501-duke-vit-bnneck:v2")
        self.assertEqual(config["release"]["base_release_id"], "market1501-vit-bnneck:v1-dvc")
        self.assertEqual(config["data"]["dataset"]["name"], "dukemtmc-reid")
        self.assertEqual(config["data"]["expected_train_images"], 16522)
        self.assertEqual(config["data"]["expected_train_identities"], 702)
        self.assertEqual(config["transfer"]["expected_checkpoint_epoch"], 48)
        self.assertTrue(config["transfer"]["reset_classifier_heads"])
        self.assertFalse(config["model"]["pretrained"])
        self.assertEqual(config["data"]["batch_size"], 16)
        self.assertEqual(config["data"]["eval_batch_size"], 16)
        self.assertEqual(config["train"]["epochs"], 50)
        self.assertEqual(config["train"]["progressive_unfreeze"]["freeze_lower_blocks"], 8)
        self.assertEqual(config["train"]["progressive_unfreeze"]["full_unfreeze_epoch"], 6)
        self.assertEqual(config["train"]["backbone_lr_factor"], 0.05)
        self.assertEqual(config["train"]["warmup_epochs"], 3)
        self.assertEqual(config["train"]["scheduler_type"], "cosine")
        self.assertTrue(config["train"]["early_stopping"]["enabled"])
        self.assertEqual(config["train"]["early_stopping"]["patience"], 5)
        self.assertEqual(config["evaluation"]["query_batch_size"], 64)

    def test_progressive_phase_freezes_only_lower_vit_blocks(self) -> None:
        model = FakeModel()
        phase = set_vit_lower_blocks_trainable(model, frozen_lower_blocks=8)
        self.assertEqual(
            phase,
            {"total_blocks": 12, "frozen_lower_blocks": 8, "trainable_upper_blocks": 4},
        )
        self.assertFalse(next(model.backbone.encoder.layers[0].parameters()).requires_grad)
        self.assertTrue(next(model.backbone.encoder.layers[11].parameters()).requires_grad)
        self.assertFalse(next(model.backbone.conv_proj.parameters()).requires_grad)
        self.assertFalse(model.backbone.class_token.requires_grad)
        self.assertFalse(model.backbone.encoder.pos_embedding.requires_grad)

    def test_classifier_parameters_are_recognized_as_task_heads(self) -> None:
        self.assertTrue(is_task_head_parameter("classifier.weight"))
        self.assertTrue(is_task_head_parameter("auxiliary_classifiers.0.weight"))
        self.assertFalse(is_task_head_parameter("backbone.encoder.layers.encoder_layer_0.mlp.0.weight"))

    def test_preflight_accepts_approved_complete_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = root / "decision.json"
            checkpoint = root / "best_model.pth"
            dataset = root / "dukemtmc-reid"
            (dataset / "bounding_box_train").mkdir(parents=True)
            (dataset / "query").mkdir()
            (dataset / "bounding_box_test").mkdir()
            decision.write_text(
                json.dumps(
                    {
                        "status": "RETRAIN_APPROVED",
                        "base_release_id": "market1501-vit-bnneck:v1-dvc",
                        "target_dataset": "dukemtmc-reid",
                        "target_dataset_version": "dukemtmc-reid",
                        "target_epochs": 50,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint.write_bytes(b"checkpoint")
            train_images = [
                Path(f"{index % 702:04d}_c{index % 8 + 1}_f{index:07d}.jpg")
                for index in range(16522)
            ]

            with patch("src.validate_duke_v2_training.shutil.disk_usage") as disk_usage, patch(
                "src.validate_duke_v2_training.Path.glob", return_value=train_images
            ):
                disk_usage.return_value.free = MINIMUM_FREE_BYTES
                result = validate_inputs(
                    decision,
                    checkpoint,
                    dataset,
                    root,
                    ROOT / "configs/duke_v2_train.yaml",
                    require_cuda=False,
                )

            self.assertEqual(result["decision"], "RETRAIN_APPROVED")
            self.assertEqual(result["target_epochs"], 50)
            self.assertEqual(result["train_images"], 16522)
            self.assertEqual(result["train_identities"], 702)
            self.assertEqual(result["train_cameras"], 8)

    def test_local_training_script_has_valid_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/train_duke_v2_local.sh")],
            check=True,
        )
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/evaluate_duke_v2.sh")],
            check=True,
        )

    def test_preflight_rejects_epoch_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = root / "decision.json"
            checkpoint = root / "best_model.pth"
            dataset = root / "dukemtmc-reid"
            config = root / "config.yaml"
            dataset.mkdir()
            decision.write_text(
                json.dumps(
                    {
                        "status": "RETRAIN_APPROVED",
                        "base_release_id": "market1501-vit-bnneck:v1-dvc",
                        "target_dataset": "dukemtmc-reid",
                        "target_dataset_version": "dukemtmc-reid",
                        "target_epochs": 10,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint.write_bytes(b"checkpoint")
            config.write_text(
                (ROOT / "configs/duke_v2_train.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Approved decision mismatch"):
                validate_inputs(
                    decision,
                    checkpoint,
                    dataset,
                    root,
                    config,
                    require_cuda=False,
                )

    def test_preflight_rejects_changed_progressive_unfreeze_contract(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/duke_v2_train.yaml").read_text(encoding="utf-8")
        )
        config["train"]["progressive_unfreeze"]["freeze_lower_blocks"] = 6
        decision = {
            "status": "RETRAIN_APPROVED",
            "base_release_id": "market1501-vit-bnneck:v1-dvc",
            "target_dataset": "dukemtmc-reid",
            "target_dataset_version": "dukemtmc-reid",
            "target_epochs": 50,
        }
        from src.validate_duke_v2_training import validate_training_contract

        with self.assertRaisesRegex(ValueError, "freeze_lower_blocks"):
            validate_training_contract(config, decision)

    def test_preflight_accepts_immutable_inbox_revision_separate_from_dataset_name(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/duke_v2_train.yaml").read_text(encoding="utf-8")
        )
        revision = "9" * 64
        config["data"]["dataset"]["version"] = revision
        decision = {
            "status": "RETRAIN_APPROVED",
            "base_release_id": "market1501-vit-bnneck:v1-dvc",
            "target_dataset": "dukemtmc-reid",
            # Existing decisions use the logical dataset name here.
            "target_dataset_version": "dukemtmc-reid",
            "target_epochs": 50,
        }
        from src.validate_duke_v2_training import validate_training_contract

        validate_training_contract(
            config,
            decision,
            expected_dataset_revision=revision,
        )

    def test_preflight_rejects_changed_inbox_revision(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/duke_v2_train.yaml").read_text(encoding="utf-8")
        )
        config["data"]["dataset"]["version"] = "a" * 64
        decision = {
            "base_release_id": "market1501-vit-bnneck:v1-dvc",
            "target_dataset": "dukemtmc-reid",
            "target_dataset_version": "dukemtmc-reid",
            "target_epochs": 50,
        }
        from src.validate_duke_v2_training import validate_training_contract

        with self.assertRaisesRegex(ValueError, "immutable revision"):
            validate_training_contract(
                config,
                decision,
                expected_dataset_revision="b" * 64,
            )


if __name__ == "__main__":
    unittest.main()
