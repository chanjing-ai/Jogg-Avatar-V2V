import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from Avatar.utils.args_config import _resolve_placeholders
from Avatar.utils.inference_validation import (
    sidecar_paths,
    split_model_paths,
    validate_job,
    validate_runtime_config,
)
from Avatar.utils.io_utils import loop_video_index
from script.export_checkpoint import normalize_state_dict


class ReleaseSmokeTest(unittest.TestCase):
    def test_config_placeholders_support_environment_defaults(self):
        config = {
            "model_dir": "${JOGG_AVATAR_MODEL_DIR:-models}",
            "vae_path": "${model_dir}/base/vae.pth",
        }
        self.assertEqual(
            _resolve_placeholders(config)["vae_path"], "models/base/vae.pth"
        )
        with mock.patch.dict(
            "os.environ", {"JOGG_AVATAR_MODEL_DIR": "/opt/jogg-models"}
        ):
            self.assertEqual(
                _resolve_placeholders(config)["vae_path"],
                "/opt/jogg-models/base/vae.pth",
            )

    def test_folded_model_paths_are_trimmed(self):
        self.assertEqual(
            split_model_paths("one.safetensors, two.safetensors,\n three.safetensors"),
            ["one.safetensors", "two.safetensors", "three.safetensors"],
        )

    def test_runtime_validation_reports_configuration_errors(self):
        args = SimpleNamespace(
            text_encoder_path="missing-text.pth",
            vae_path="missing-vae.pth",
            lora_ckpt_path="missing-avatar.safetensors",
            dit_path="missing-1.safetensors,missing-2.safetensors",
            wav2vec_path="missing-wav2vec",
            use_audio=True,
            sp_size=2,
            world_size=1,
            overlap_frame=12,
            num_steps=0,
            fps=0,
            sample_rate=16000,
        )
        with self.assertRaises(ValueError) as context:
            validate_runtime_config(args)
        message = str(context.exception)
        self.assertIn("missing-1.safetensors", message)
        self.assertIn("sp_size (2) must equal torchrun world size (1)", message)
        self.assertIn("overlap_frame must equal 1 + 4*n", message)
        self.assertIn("num_steps must be at least 1", message)

    def test_job_validation_requires_face_metadata_until_cache_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "clip.mp4"
            audio = Path(tmpdir) / "drive.wav"
            video.touch()
            audio.touch()
            mouth_info, latent = sidecar_paths(str(video))
            with self.assertRaisesRegex(FileNotFoundError, "extract_mouth_info"):
                validate_job(str(video), str(audio), True)

            Path(mouth_info).write_text(
                json.dumps({"1": {"face_bbox": [0, 10, 0, 10]}}),
                encoding="utf-8",
            )
            self.assertEqual(
                validate_job(str(video), str(audio), True),
                (mouth_info, latent),
            )

    def test_bounce_index_never_leaves_source_range(self):
        self.assertEqual(
            [loop_video_index(index, 4) for index in range(10)],
            [0, 1, 2, 3, 2, 1, 0, 1, 2, 3],
        )

    def test_checkpoint_export_normalizes_lightning_prefix(self):
        tensor = torch.ones(2, 3)
        result = normalize_state_dict(
            {"pipe.dit.blocks.0.weight": tensor, "global_step": 10}
        )
        self.assertEqual(list(result), ["blocks.0.weight"])
        self.assertTrue(result["blocks.0.weight"].is_contiguous())


if __name__ == "__main__":
    unittest.main()

