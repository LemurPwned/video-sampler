from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class SaveFormatConfig(BaseModel):
    """
    Configuration options for the save format.

    Args:
        encode_time_b64 (bool, optional): Encode time as base64 string. Avoids `.`
            in the filename. Defaults to False.
        avoid_dot (bool, optional): Avoid `.` in the filename by replacing
            it with `_`. Defaults to False.
        include_filename (bool, optional): Include filename in the output
            path. Defaults to False.
    """

    encode_time_b64: bool = Field(default=False)
    avoid_dot: bool = Field(default=False)
    include_filename: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_save_format(cls, values: "SaveFormatConfig"):
        if values.encode_time_b64 and values.avoid_dot:
            import warnings

            warnings.warn(
                "encode_time_b64 and avoid_dot are both True, "
                "avoid_dot will be ignored",
                UserWarning,
                stacklevel=2,
            )
        return values


class SamplerConfig(BaseModel):
    """
    Configuration options for the video sampler.

    Args:
        min_frame_interval_sec (float, optional): The minimum time interval
            between sampled frames in seconds. Defaults to 1.
        keyframes_only (bool, optional): Flag indicating whether to
            sample only keyframes. Defaults to True.
        queue_wait (float, optional): The time to wait between checking
            the frame queue in seconds. Defaults to 0.1.
        start_time_s (int, optional): The starting time for sampling in seconds.
            Defaults to 0.
        end_time_s (int, optional): The ending time for sampling in seconds.
            None for no end. Defaults to None.
        precise_seek (bool, optional): Flag indicating whether to use precise
            seeking. Defaults to False. Precise seeking is slower but more
            accurate.
        debug (bool, optional): Flag indicating whether to enable debug mode.
            Defaults to False.
        print_stats (bool, optional): Flag indicating whether to print
            sampling statistics. Defaults to False.
        buffer_config (dict[str, Any], optional): Configuration options for
                the frame buffer. Defaults to {"type": "entropy", "size": 15,
                "debug": True}.
        gate_config (dict[str, Any], optional): Configuration options for
                the frame gate. Defaults to {"type": "pass"}.
        extractor_config (dict[str, Any], optional): Configuration options for
                the extractor (keyword, audio). Defaults to None.
        summary_config (dict[str, Any], optional): Configuration options for
                the summary generator. Defaults to None.
    Methods:
        __str__() -> str:
            Returns a string representation of the configuration.

    """

    min_frame_interval_sec: float = Field(default=1, ge=0)
    keyframes_only: bool = Field(default=True)
    queue_wait: float = Field(default=0.1, ge=1e-3)
    start_time_s: int = Field(
        default=0, ge=0, description="The starting time for sampling in seconds."
    )
    end_time_s: int | None = Field(
        default=None,
        ge=1,
        description="The ending time for sampling in seconds. None for no end.",
    )
    precise_seek: bool = Field(default=False, description="Use precise seeking.")
    debug: bool = Field(default=False)
    print_stats: bool = Field(default=False)
    buffer_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "hash",
            "hash_size": 8,
            "size": 15,
            "debug": True,
        }
    )
    gate_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "pass",
        }
    )
    extractor_config: dict[str, Any] = Field(default_factory=dict)
    summary_config: dict[str, Any] = Field(default_factory=dict)
    n_workers: int = Field(default=1)

    save_format: SaveFormatConfig = Field(default_factory=SaveFormatConfig)

    def __str__(self) -> str:
        return str(self.model_dump())

    @classmethod
    def from_yaml(cls, file_path: str) -> "SamplerConfig":
        with open(file_path) as file:
            data = yaml.safe_load(file)
        return cls(**data)

    @model_validator(mode="after")
    def validate_start_end_times(cls, values: "SamplerConfig"):
        if values.end_time_s is not None and values.start_time_s >= values.end_time_s:
            raise ValueError("start_time_s must be strictly less than the end_time_s")
        return values
