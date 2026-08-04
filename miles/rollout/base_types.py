from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState


@dataclass(frozen=True)
class RolloutFnConstructorInput:
    args: Namespace
    # TODO may refactor DataSource API
    data_source: DataSource


@dataclass(frozen=True)
class RolloutFnBaseInput:
    rollout_id: int

    @property
    def evaluation(self):
        raise NotImplementedError


# subclassing for different data in the future
@dataclass(frozen=True)
class RolloutFnTrainInput(RolloutFnBaseInput):
    @property
    def evaluation(self):
        return False


@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    @property
    def evaluation(self):
        return True


# TODO make it frozen
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


# TODO make it frozen
@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


RolloutFnInput = RolloutFnTrainInput | RolloutFnEvalInput
RolloutFnOutput = RolloutFnTrainOutput | RolloutFnEvalOutput


@dataclass(frozen=True)
class GenerateFnInput:
    state: GenerateState
    sample: Sample
    sampling_params: dict[str, Any]
    evaluation: bool

    @property
    def args(self) -> Namespace:
        return self.state.args


@dataclass(frozen=True)
class GenerateFnOutput:
    # One generate may lead to multiple samples, such as multi-agent, tree-like exploration, or
    # multi-turn with removing thinking tokens.
    samples: Sample | list[Sample]


def call_rollout_fn(fn, *args, evaluation: bool, rlix_hooks=None, **kwargs):
    """Legacy rollout function call interface. Used when MILES_EXPERIMENTAL_ROLLOUT_REFACTOR is disabled.

    ``rlix_hooks``: passed through to the underlying rollout function as a
    keyword argument when the function accepts it (the canonical
    ``generate_rollout_fully_async`` and ``generate_rollout_async`` entries
    do). Without this, the rollout function falls back to
    :class:`NoOpRLixHooks` and every ``begin_progress_batch`` /
    ``bump_completed`` call is a silent no-op — the central scheduler then
    has no demand signal between rollouts, so its gap-ratio planner cannot
    wake engines for rollout N+1 after rollout N's ``_after_training``.
    """
    import inspect as _inspect

    fn_params = _inspect.signature(fn).parameters
    if rlix_hooks is not None and "rlix_hooks" in fn_params:
        kwargs = {**kwargs, "rlix_hooks": rlix_hooks}
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output


class EnginePreemptedError(Exception):
    """Raised when the active inference engine has been preempted by the RLix scheduler.

    Caught at multi_turn boundaries for turn-level redispatch, and surfaced to the
    fully_async output queue as a ``_FatalError`` sentinel after the redispatch attempt
    cap is exhausted.
    """


class RLixRouterMetadataError(Exception):
    """Raised when an RLix-mode generate response is missing router-injected metadata.

    The MILES router injects ``meta_info["miles_admission_disabled"]``
    into every ``/generate`` JSON response in RLix mode. Absence is treated as a fatal
    misconfiguration rather than allowing turn-level redispatch to silently degrade.
    """
