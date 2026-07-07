"""Versioned model-turn metadata accepted from an Aviary resources server."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class ModelTokenUsage(_WireModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ModelTurn(_WireModel):
    response_id: str = Field(min_length=1)
    turn_index: int = Field(ge=1)
    usage: ModelTokenUsage | None = None


class NemoGymStepContextV1(_WireModel):
    version: Literal[1]
    model_turns: list[ModelTurn] = Field(min_length=1)


def model_turns_from_action_info(info: dict[str, Any] | None) -> list[ModelTurn]:
    """Read the optional NeMo Gym envelope without coupling Hypotest to Gym."""
    if info is None or "nemo_gym" not in info:
        return []

    nemo_gym = info["nemo_gym"]
    if not isinstance(nemo_gym, dict):
        raise TypeError("action info field 'nemo_gym' must be an object")
    if "step_context" not in nemo_gym:
        return []

    return NemoGymStepContextV1.model_validate(nemo_gym["step_context"]).model_turns
