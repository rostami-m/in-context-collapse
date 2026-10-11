import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from peft import LoraConfig, get_peft_model
import torch

from incontext_collapse.continual import cl_layer_bands, cl_target_manifest, main
from incontext_collapse.core import region_targets


class QwenLayout(torch.nn.Module):
    """Dimension-faithful module names without allocating real model weights."""

    def __init__(self, n_layers=28, device="meta", small=False):
        super().__init__()
        self.config = {"model_type": "qwen2_vl"}
        hidden, kv, intermediate, merged = (4, 2, 8, 8) if small else (1536, 256, 8960, 5120)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            for name, output in (("q_proj", hidden), ("k_proj", kv),
                                 ("v_proj", kv), ("o_proj", hidden)):
                setattr(layer.self_attn, name, torch.nn.Linear(hidden, output, device=device))
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_proj = torch.nn.Linear(hidden, intermediate, device=device)
            layer.mlp.up_proj = torch.nn.Linear(hidden, intermediate, device=device)
            layer.mlp.down_proj = torch.nn.Linear(intermediate, hidden, device=device)
            self.model.layers.append(layer)
        self.visual = torch.nn.Module()
        self.visual.merger = torch.nn.Module()
        self.visual.merger.mlp = torch.nn.Sequential(
            torch.nn.Linear(merged, merged, device=device), torch.nn.GELU(),
            torch.nn.Linear(merged, hidden, device=device))


class ContinualTargetTests(unittest.TestCase):
    def test_explicit_partition_is_disjoint_and_exhaustive(self):
        bands = cl_layer_bands(28)
        self.assertEqual(bands, {"early": list(range(10)), "mid": list(range(10, 19)),
                                 "late": list(range(19, 28))})
        self.assertEqual(sum(bands.values(), []), list(range(28)))
        self.assertEqual([len(v) for v in cl_layer_bands(6).values()], [2, 2, 2])
        with self.assertRaises(ValueError):
            cl_layer_bands(2)

    def test_published_capacities_match_resolved_and_installed_modules(self):
        expected = {"circuit": (135, 6400000), "all": (196, 9232384),
                    "late": (63, 2967552)}
        for region, (n_targets, count) in expected.items():
            with self.subTest(region=region):
                model = QwenLayout()
                manifest = cl_target_manifest(model, region)
                self.assertEqual(manifest["n_targets"], n_targets)
                self.assertEqual(manifest["trainable_parameters"], count)
                self.assertEqual(len(set(manifest["target_modules"])), n_targets)
                fitted = get_peft_model(model, LoraConfig(
                    r=8, lora_alpha=16, target_modules=manifest["target_modules"],
                    lora_dropout=0.05))
                self.assertEqual(sum(p.numel() for p in fitted.parameters() if p.requires_grad), count)

    def test_primary_region_helper_is_not_reused_or_changed(self):
        model = QwenLayout()
        primary_late = region_targets(model, "late")
        cl_late = cl_target_manifest(model, "late")["target_modules"]
        self.assertEqual(len(primary_late), 70)
        self.assertEqual(len(cl_late), 63)
        self.assertTrue(any(name.startswith("model.layers.18.") for name in primary_late))
        self.assertFalse(any(name.startswith("model.layers.18.") for name in cl_late))

    def test_manifest_is_json_serializable_and_counts_dimensions(self):
        manifest = cl_target_manifest(QwenLayout(), "circuit")
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        self.assertEqual(manifest["selected_decoder_layers"], list(range(19)))
        self.assertTrue(manifest["connector_included"])
        for target in manifest["target_shapes"]:
            self.assertEqual(target["lora_parameters"],
                             8 * (target["in_features"] + target["out_features"]))

    def test_unrecognized_model_or_missing_projection_fails_closed(self):
        model = QwenLayout()
        model.config["model_type"] = "other"
        with self.assertRaisesRegex(ValueError, "Qwen2-VL"):
            cl_target_manifest(model, "all")
        model.config["model_type"] = "qwen2_vl"
        del model.model.layers[0].mlp.up_proj
        with self.assertRaisesRegex(ValueError, "missing or is not linear"):
            cl_target_manifest(model, "all")
        with self.assertRaises(ValueError):
            cl_target_manifest(model, "unknown")

    def test_existing_output_is_rejected_before_loading_data_or_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preserved.json"
            path.write_text("retained evidence")
            with patch("sys.argv", ["continual", "--out", str(path)]), \
                    patch("incontext_collapse.continual.cl_tasks") as tasks, \
                    patch("transformers.AutoProcessor.from_pretrained") as processor, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            tasks.assert_not_called()
            processor.assert_not_called()
            self.assertEqual(path.read_text(), "retained evidence")

    def test_new_run_persists_exact_target_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new" / "result.json"
            argv = ["continual", "--T", "1", "--seeds", "1", "--steps", "1",
                    "--regions", "circuit", "--out", str(path)]
            with patch("sys.argv", argv), \
                    patch("incontext_collapse.continual.pick_device", return_value="cpu"), \
                    patch("incontext_collapse.continual.cl_tasks", return_value=[({}, [])]), \
                    patch("incontext_collapse.continual.label_token_ids", return_value={}), \
                    patch("incontext_collapse.continual.train_task") as trainer, \
                    patch("incontext_collapse.continual.accuracy", return_value=0.75), \
                    patch("transformers.AutoProcessor.from_pretrained"), \
                    patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained",
                          return_value=QwenLayout(3, device="cpu", small=True)), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads(path.read_text())
            manifest = result["circuit"]["target_manifest"]
            self.assertEqual(result["run_config"]["target_partition"], manifest["partition"])
            self.assertEqual(result["circuit"]["trainable_params"], manifest["trainable_parameters"])
            self.assertEqual(manifest["selected_decoder_layers"], [0, 1])
            self.assertEqual(result["circuit"]["per_seed"][0]["matrix"], [[0.75]])
            trainer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
