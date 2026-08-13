import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Avatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg


class AudioPreprocessTest(unittest.TestCase):
    def test_zero_silence_copies_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.wav"
            output = Path(tmpdir) / "output.wav"
            source.write_bytes(b"audio")
            with mock.patch("subprocess.run") as run:
                result = add_silence_to_audio_ffmpeg(source, output, 0)
            run.assert_not_called()
            self.assertEqual(Path(result), output)
            self.assertEqual(output.read_bytes(), b"audio")

    def test_positive_silence_uses_bounded_delay_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.wav"
            output = Path(tmpdir) / "output.wav"
            source.touch()
            with mock.patch("subprocess.run") as run:
                add_silence_to_audio_ffmpeg(source, output, 0.3)
            command = run.call_args.args[0]
            self.assertIn("adelay=300:all=1", command)
            command_text = " ".join(map(str, command))
            self.assertNotIn("apad", command_text)
            self.assertNotIn("anullsrc", command_text)
            run.assert_called_once_with(command, check=True)

    def test_negative_silence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            add_silence_to_audio_ffmpeg("source.wav", "output.wav", -0.1)


if __name__ == "__main__":
    unittest.main()
