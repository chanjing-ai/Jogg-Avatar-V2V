import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from examples.wanvideo.train_wan2_2_5b import (
    REQUIRED_FEATURE_KEYS,
    Wan22FeatureDataset,
    _training_state_dict,
)


class TrainingDataTest(unittest.TestCase):
    def test_feature_dataset_loads_preprocessed_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = root / "all.csv"
            metadata.write_text("file_name\nclip.mp4\n", encoding="utf-8")
            feature_path = root / "clip.mp4.tensors.vae2.2.pth"
            torch.save({key: torch.zeros(1) for key in REQUIRED_FEATURE_KEYS}, feature_path)

            dataset = Wan22FeatureDataset(str(root), str(metadata), steps_per_epoch=3)
            self.assertEqual(len(dataset), 3)
            sample = dataset[2]
            self.assertTrue(REQUIRED_FEATURE_KEYS.issubset(sample))
            self.assertIn(sample["pre_fix_frames_num"], (13, -13))

    def test_metadata_requires_file_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = root / "all.csv"
            pd.DataFrame({"text": ["hello"]}).to_csv(metadata, index=False)
            with self.assertRaisesRegex(ValueError, "file_name"):
                Wan22FeatureDataset(str(root), str(metadata), None)

    def test_release_weights_can_resume_training(self):
        tensor = torch.ones(1)
        result = _training_state_dict(
            {
                "blocks.0.self_attn.q.lora_A.default.weight": tensor,
                "audio_proj.proj.weight": tensor,
            },
            {
                "pipe.dit.blocks.0.self_attn.q.lora_A.default.weight",
                "audio_proj.proj.weight",
            },
        )
        self.assertIn(
            "pipe.dit.blocks.0.self_attn.q.lora_A.default.weight", result
        )
        self.assertIn("audio_proj.proj.weight", result)


if __name__ == "__main__":
    unittest.main()
