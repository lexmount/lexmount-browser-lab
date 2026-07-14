"""Container-level regression test for LexBrowser's multi-turn GRPO adapter."""

from __future__ import annotations

from copy import deepcopy

from lexbrowser_nemo_gym_grouping import _postprocess_with_stable_group_prompt
from nemo_rl.environments.nemo_gym import NemoGym


class _Tokenizer:
    def batch_decode(self, batches):
        return ["/".join(map(str, batch)) for batch in batches]

    def apply_chat_template(self, messages, tokenize=True):
        return [0]


def _result(observation_token: int) -> dict:
    return {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [10, 11],
                    "generation_token_ids": [12],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [10, 11, 12, observation_token],
                    "generation_token_ids": [14],
                    "generation_log_probs": [-0.2],
                },
            ]
        },
        "responses_create_params": {"input": []},
    }


def _grpo_prompt_tokens(message_log):
    return tuple(
        token
        for message in message_log
        if message["role"] in {"user", "system"}
        for token in message["token_ids"].tolist()
    )


def main() -> None:
    tokenizer = _Tokenizer()
    first = _postprocess_with_stable_group_prompt(None, deepcopy(_result(13)), tokenizer)
    second = _postprocess_with_stable_group_prompt(None, deepcopy(_result(99)), tokenizer)

    assert [message["role"] for message in first["message_log"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert _grpo_prompt_tokens(first["message_log"]) == (10, 11)
    assert _grpo_prompt_tokens(first["message_log"]) == _grpo_prompt_tokens(
        second["message_log"]
    )

    metadata = getattr(NemoGym, "__ray_metadata__", None)
    target = getattr(metadata, "modified_class", NemoGym)
    assert getattr(target, "_lexbrowser_stable_group_prompt", False)
    assert target._postprocess_nemo_gym_to_nemo_rl_result is _postprocess_with_stable_group_prompt
    print("PASS: stable GRPO prompt identity preserves multi-turn tool context")


if __name__ == "__main__":
    main()
