---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B
pipeline_tag: video-to-video
tags:
  - audio-driven
  - avatar
  - talking-head
  - video-to-video
  - wan2.2
---

# Jogg-Avatar V2V 5B

Jogg-Avatar V2V is an audio-driven video-to-video avatar model based on
Wan2.2-TI2V-5B. It preserves the source video's body, camera, and background
motion while regenerating the face region to follow a driving audio track.

Training and inference code is available at
[chanjing-ai/Jogg-Avatar-V2V](https://github.com/chanjing-ai/Jogg-Avatar-V2V).

Use these weights with the Jogg-Avatar-V2V code repository. The model directory
must contain:

```text
Jogg-Avatar-Wan2.2-5B/
|-- config.json
`-- diffusion_pytorch_model.safetensors
```

The Wan2.2 base model and `facebook/wav2vec2-base-960h` are required separately.
Review their licenses and terms before use. Users are responsible for obtaining
consent for source videos and voices and for clearly disclosing synthetic media.
