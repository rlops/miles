"""
Simple multi-turn generation with tool calling.

F3 / F29 / F30 / F31 turn-level redispatch: when the F3 router metadata
classifies a /generate response as a scheduler preempt
(``meta_info["miles_admission_disabled"] == True``), restore the pre-turn
sample state and try the same turn against a different engine. The
attempt cap is the *total* engine count
(``args.rollout_num_gpus // args.rollout_num_gpus_per_engine``), not the
active count, so a shrink-to-1 cycle still gets a fair retry budget. On
exhaustion raise :class:`EnginePreemptedError` (caught at the
fully_async ``_FatalError`` queue boundary in iter 16).
"""

import argparse
from copy import deepcopy

from miles.rollout.base_types import (
    EnginePreemptedError,
    GenerateFnInput,
    GenerateFnOutput,
    RLixRouterMetadataError,
)
from miles.rollout.generate_utils.generate_endpoint_utils import (
    _restore_turn_state,
    _snapshot_turn_state,
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.rlix_validation import is_rlix_mode


def _is_scheduler_preempt(output: dict, *, rlix_mode: bool) -> bool:
    """Classify a /generate response as a scheduler preempt.

    Standalone (rlix_mode=False) always returns False — preempt classification
    is RLix-only.

    RLix mode reads ``output["meta_info"]["miles_admission_disabled"]``. If
    the metadata fields are missing entirely, raise
    :class:`RLixRouterMetadataError` (rather than silently classifying the
    response as non-preempt) so misconfiguration surfaces immediately.
    """
    if not rlix_mode:
        return False
    meta = output.get("meta_info") if isinstance(output, dict) else None
    if not isinstance(meta, dict) or "miles_admission_disabled" not in meta:
        raise RLixRouterMetadataError(
            "RLix mode /generate response is missing "
            "meta_info['miles_admission_disabled']; check that the router "
            "is the MILES admission router (not stock sglang_router) and "
            "that response-body mutation is path-guarded for /generate."
        )
    return bool(meta["miles_admission_disabled"])


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # ----------------------- Setup -------------------------

    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    # multi_turn.generate has never implemented partial-rollout resume
    # semantics (no `len(sample.response) > 0` short-circuit like
    # single_turn.py); prior responses would be silently re-tokenized as
    # fresh prompts. The unconditional assert preserves the long-standing
    # standalone safety guard. Under RLix mode the same constraint stands
    # (F29 / C17): radix middleware is off, turn-level redispatch
    # requires non-streaming JSON, and partial_rollout has no place in
    # either mode.
    rlix_mode = is_rlix_mode()
    assert not args.partial_rollout, (
        "Partial rollout is not supported in multi_turn.generate (F29 / C17)"
    )

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    execute_tool_function = load_function(args.generate_execute_tool_function_path)

    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)

    multi_samples = []

    # ----------------------- Initial prompts -------------------------

    prompt_tokens_ids = compute_prompt_ids_from_sample(input.state, sample, tools=tool_specs)

    sample.tokens = prompt_tokens_ids.copy()

    # F29 redispatch attempt cap. Use total engine count (not active count)
    # so a shrink-to-1 cycle still gets a fair retry budget.
    per_engine = max(int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1), 1)
    total_engines = max(int(getattr(args, "rollout_num_gpus", 0) or 0) // per_engine, 1)
    max_redispatch_attempts = total_engines

    for _turn in range(args.generate_max_turns):
        # ----------------------- Call inference endpoint -------------------------

        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            sample.status = halt_status
            if args.generate_multi_samples and multi_samples:
                multi_samples[-1].status = halt_status
            break

        # F32 metadata injection requires a JSON body — force stream=False
        # under RLix mode only. Standalone keeps its pre-existing payload
        # shape so existing exact-payload tests are unaffected.
        if rlix_mode and isinstance(payload, dict):
            payload["stream"] = False

        if args.generate_multi_samples:
            sample = deepcopy(input.sample)

        # F29 turn-level redispatch loop: snapshot pre-turn state, post,
        # classify, restore-on-preempt up to max_redispatch_attempts.
        snapshot = _snapshot_turn_state(sample, multi_samples)
        attempt = 0
        while True:
            output = await post(url, payload)
            if not _is_scheduler_preempt(output, rlix_mode=rlix_mode):
                break
            attempt += 1
            if attempt >= max_redispatch_attempts:
                raise EnginePreemptedError(
                    f"turn-level redispatch budget exhausted "
                    f"({attempt}/{max_redispatch_attempts}); engines remained "
                    f"admission-closed for the entire pool"
                )
            _restore_turn_state(sample, multi_samples, snapshot)

        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)

        if args.generate_multi_samples:
            multi_samples.append(deepcopy(sample))

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            break

        # ----------------------- Execute tools -------------------------

        _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        if len(tool_calls) == 0:
            break

        tool_messages = await execute_tool_calls(tool_calls, execute_tool_function)
        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)

    return GenerateFnOutput(samples=multi_samples if args.generate_multi_samples else sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")


generate.add_arguments = _add_arguments
