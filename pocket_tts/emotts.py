"""Optional inference-time emotion steering for PocketTTS."""

import copy
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from typing_extensions import Self

from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_SAMPLER_DECODE_STEPS,
)
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.utils import download_if_necessary

EMOTION_PROSODY_DEFAULTS = {
    "angry": {"temp": 1.3, "noise_clamp": 3.0, "lsd_decode_steps": 1},
    "disgust": {"temp": 0.5, "noise_clamp": 1.5, "lsd_decode_steps": 2},
    "fear": {"temp": 1.6, "noise_clamp": None, "lsd_decode_steps": 1},
    "happy": {"temp": 0.9, "noise_clamp": 2.5, "lsd_decode_steps": 2},
    "neutral": {"temp": 0.7, "noise_clamp": None, "lsd_decode_steps": 1},
    "sad": {"temp": 0.4, "noise_clamp": 0.8, "lsd_decode_steps": 3},
}


def _profile(overrides: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    result = {name: values.copy() for name, values in EMOTION_PROSODY_DEFAULTS.items()}
    if overrides is None:
        return result
    if not isinstance(overrides, dict):
        raise TypeError("prosody_overrides must be a dictionary")
    for name, values in overrides.items():
        if name not in result:
            raise ValueError(f"Unknown emotion: {name}")
        if not isinstance(values, dict) or set(values) != {
            "temp",
            "noise_clamp",
            "lsd_decode_steps",
        }:
            raise ValueError(f"Invalid prosody override for {name}")
        temp, clamp, steps = (values["temp"], values["noise_clamp"], values["lsd_decode_steps"])
        if (
            isinstance(temp, bool)
            or not isinstance(temp, (int, float))
            or not math.isfinite(temp)
            or temp <= 0
        ):
            raise ValueError("temp must be positive and finite")
        if clamp is not None and (
            isinstance(clamp, bool)
            or not isinstance(clamp, (int, float))
            or not math.isfinite(clamp)
            or clamp <= 0
        ):
            raise ValueError("noise_clamp must be positive and finite or None")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("lsd_decode_steps must be a positive integer")
        result[name] = {"temp": float(temp), "noise_clamp": clamp, "lsd_decode_steps": steps}
    return result


def _load_vectors(source: str | Path) -> dict[str, torch.Tensor]:
    source = str(source)
    if source.startswith(("http://", "https://")):
        raise ValueError("Use a local weights file or a revision-pinned hf:// URL")
    if source.startswith("hf://") and "@" not in source:
        raise ValueError("Hugging Face steering weights require a pinned @revision")
    path = download_if_necessary(source)
    if path.suffix == ".safetensors":
        state = load_file(str(path))
    elif path.suffix in {".pt", ".pth"}:
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("Steering weights must be .safetensors, .pt, or .pth")
    if not isinstance(state, dict):
        raise TypeError("Steering weights must be a tensor dictionary")
    prefix = "steering_vectors."
    vectors = {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }
    if set(vectors) != set(EMOTION_PROSODY_DEFAULTS):
        raise ValueError("Steering weights must contain all six emotion vectors")
    if any(not isinstance(v, torch.Tensor) or v.ndim != 1 for v in vectors.values()):
        raise ValueError("Steering vectors must be one-dimensional tensors")
    if len({v.numel() for v in vectors.values()}) != 1:
        raise ValueError("Steering vectors must have equal dimensions")
    return vectors


class EmoShiftLayer(nn.Module):
    EMOTIONS = tuple(EMOTION_PROSODY_DEFAULTS)

    def __init__(self, vectors: dict[str, torch.Tensor], injection_layer: int = 5):
        super().__init__()
        self.injection_layer = injection_layer
        self.steering_vectors = nn.ParameterDict(
            {
                name: nn.Parameter(value.detach().clone(), requires_grad=False)
                for name, value in vectors.items()
            }
        )
        self.active_emotion = "neutral"
        self.intensity = 0.0
        self._cache: tuple[tuple[str, torch.device, torch.dtype], torch.Tensor] | None = None

    def set_emotion(self, emotion: str, intensity: float) -> None:
        if emotion not in self.EMOTIONS:
            raise ValueError(f"Unknown emotion: {emotion}")
        if (
            isinstance(intensity, bool)
            or not isinstance(intensity, (int, float))
            or not math.isfinite(intensity)
            or intensity < 0
        ):
            raise ValueError("intensity must be finite and non-negative")
        self.active_emotion = emotion
        self.intensity = float(intensity)
        self._cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_emotion == "neutral" or self.intensity == 0:
            return x
        key = (self.active_emotion, x.device, x.dtype)
        if self._cache is None or self._cache[0] != key:
            vector = self.steering_vectors[self.active_emotion]
            if vector.numel() != x.shape[-1]:
                raise ValueError("Steering vector dimension does not match transformer output")
            self._cache = (key, vector.to(device=x.device, dtype=x.dtype).view(1, 1, -1))
        return x + self.intensity * self._cache[1]


class EmoTTS(nn.Module):
    """Owns an isolated TTSModel while delegating its inference methods."""

    def __init__(
        self,
        tts_model: TTSModel,
        weights_path: str | Path,
        prosody_overrides: dict[str, dict[str, Any]] | None = None,
        injection_layer: int = 5,
        *,
        _owns_model: bool = False,
    ):
        super().__init__()
        profile = _profile(prosody_overrides)
        vectors = _load_vectors(weights_path)
        model = tts_model if _owns_model else copy.deepcopy(tts_model)
        layers = model.flow_lm.transformer.layers
        if (
            isinstance(injection_layer, bool)
            or not isinstance(injection_layer, int)
            or not 0 <= injection_layer < len(layers)
        ):
            raise ValueError(f"injection_layer must be between 0 and {len(layers) - 1}")
        self.model = model
        self.emo_layer = EmoShiftLayer(vectors, injection_layer).to(model.device)
        self._profile = profile
        self._base = (model.temp, model.noise_clamp, model.sampler_decode_steps)
        self._hook_handle = layers[injection_layer].register_forward_hook(self._steer)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - delegates the model API
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)

    def _steer(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401 - PyTorch hooks may receive tensor or tuple output
        if isinstance(output, tuple):
            return (self.emo_layer(output[0]), *output[1:])
        return self.emo_layer(output)

    def set_emotion(self, emotion: str | None, intensity: float = 1.0) -> None:
        emotion = "neutral" if emotion is None else emotion
        self.emo_layer.set_emotion(emotion, intensity)
        if emotion == "neutral" or intensity == 0:
            self.model.temp, self.model.noise_clamp, self.model.sampler_decode_steps = self._base
        else:
            params = self._profile[emotion]
            self.model.temp = params["temp"]
            self.model.noise_clamp = params["noise_clamp"]
            self.model.sampler_decode_steps = params["lsd_decode_steps"]

    def available_emotions(self) -> list[str]:
        return list(EmoShiftLayer.EMOTIONS)

    def cleanup(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.set_emotion(None)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()

    @classmethod
    def load_model(
        cls,
        *,
        weights_path: str | Path,
        language: str | None = None,
        config: str | Path | None = None,
        temp: float | None = None,
        sampler_decode_steps: int = DEFAULT_SAMPLER_DECODE_STEPS,
        noise_clamp: float | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
        quantize: bool = False,
        checkpoint: str | Path | None = None,
        prosody_overrides: dict[str, dict[str, Any]] | None = None,
        injection_layer: int = 5,
    ) -> "EmoTTS":
        _profile(prosody_overrides)
        _load_vectors(weights_path)
        model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temp,
            sampler_decode_steps=sampler_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
            checkpoint=checkpoint,
        )
        return cls(model, weights_path, prosody_overrides, injection_layer, _owns_model=True)
