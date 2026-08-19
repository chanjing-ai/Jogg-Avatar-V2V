from __future__ import annotations


DEFAULT_PROMPT = (
    "A realistic video of a face speaking directly to the camera. The camera "
    "remains steady and every facial detail is sharp and clearly visible."
)


def resolve_prompt(value: object) -> str:
    if value is None:
        return DEFAULT_PROMPT
    prompt = str(value).strip()
    if not prompt or prompt.lower() == "nan":
        return DEFAULT_PROMPT
    return prompt
