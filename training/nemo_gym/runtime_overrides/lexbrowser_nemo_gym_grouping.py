"""LexBrowser-only NeMo-Gym grouping adapter.

NeMo-RL v0.6 reconstructs a multi-turn Nemo-Gym trajectory as alternating
``user``/``assistant`` messages.  Its GRPO implementation then identifies a
group from all ``user`` and ``system`` tokens.  Browser observations differ
between rollouts, so treating the post-initial context as ``user`` makes every
member of an otherwise valid 8-sample GRPO group look like a distinct prompt.

This adapter preserves every token, logprob, and assistant loss mask.  It
only marks *post-initial* non-trainable context as ``tool`` instead of
``user``.  Thus GRPO's existing prompt extractor sees exactly the immutable
initial prompt while policy logprob/backward still sees the full multi-turn
context.  It is loaded at runtime for this LexBrowser experiment and does not
edit or fork NeMo-RL source files.
"""

from __future__ import annotations

from typing import Any

import torch


def _postprocess_with_stable_group_prompt(
    self: Any, nemo_gym_result: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    """Rebuild a rollout log while preserving a stable GRPO prompt identity."""
    assert isinstance(nemo_gym_result, dict), (
        f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
    )

    message_log: list[dict[str, Any]] = []
    seen_token_ids: list[int] = []
    decode_items: list[tuple[dict[str, Any], list[int], list[int]]] = []

    for output_item in nemo_gym_result["response"]["output"]:
        if "generation_token_ids" not in output_item:
            continue

        prompt_token_ids = output_item["prompt_token_ids"]
        generation_token_ids = output_item["generation_token_ids"]
        assert seen_token_ids == prompt_token_ids[: len(seen_token_ids)], (
            "Non-contiguous NeMo-Gym token stream; cannot safely reconstruct "
            "the multi-turn trajectory."
        )

        new_prompt_token_ids = prompt_token_ids[len(seen_token_ids) :]
        # Only the first prefix is the immutable task prompt.  Later prefixes
        # contain browser/tool context and must not split a GRPO group.
        prefix_role = "user" if not seen_token_ids else "tool"
        message_log.append(
            {
                "role": prefix_role,
                "content": "",
                "token_ids": torch.tensor(new_prompt_token_ids, dtype=torch.int64),
            }
        )
        message_log.append(
            {
                "role": "assistant",
                "content": "",
                "token_ids": torch.tensor(generation_token_ids, dtype=torch.int64),
                "generation_logprobs": torch.tensor(
                    output_item["generation_log_probs"]
                ),
            }
        )

        seen_token_ids.extend(new_prompt_token_ids)
        seen_token_ids.extend(generation_token_ids)
        decode_items.append((output_item, prompt_token_ids, generation_token_ids))

    if decode_items:
        prompt_texts = tokenizer.batch_decode([item[1] for item in decode_items])
        generation_texts = tokenizer.batch_decode([item[2] for item in decode_items])
        for (output_item, _, _), prompt_text, generation_text in zip(
            decode_items, prompt_texts, generation_texts
        ):
            output_item["prompt_str"] = prompt_text
            output_item["generation_str"] = generation_text
            output_item.pop("prompt_token_ids")
            output_item.pop("generation_token_ids")
            output_item.pop("generation_log_probs")

    if not message_log:
        input_messages = nemo_gym_result["responses_create_params"]["input"]
        prompt_token_ids = tokenizer.apply_chat_template(input_messages, tokenize=True)
        raise ValueError(
            "NeMo Gym returned a result with no generation data. "
            f"Prompt length: {len(prompt_token_ids)}."
        )

    return {
        "message_log": message_log,
        "input_message_log": message_log[:1],
        "full_result": nemo_gym_result,
    }


def install() -> bool:
    """Install the adapter on Ray's underlying NemoGym actor class."""
    from nemo_rl.environments.nemo_gym import NemoGym

    metadata = getattr(NemoGym, "__ray_metadata__", None)
    target = getattr(metadata, "modified_class", NemoGym)
    if getattr(target, "_lexbrowser_stable_group_prompt", False):
        return False
    target._postprocess_nemo_gym_to_nemo_rl_result = (  # type: ignore[attr-defined]
        _postprocess_with_stable_group_prompt
    )
    target._lexbrowser_stable_group_prompt = True  # type: ignore[attr-defined]
    return True
