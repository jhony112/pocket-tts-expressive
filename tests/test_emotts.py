"""Offline regression tests for optional EmoTTS steering."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from pocket_tts.emotts import EMOTION_PROSODY_DEFAULTS, EmoTTS, _profile
from pocket_tts.models.tts_model import TTSModel


class FakeTTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_lm = nn.Module()
        self.flow_lm.transformer = nn.Module()
        self._layers = nn.ModuleList([nn.Identity() for _ in range(6)])
        self.flow_lm.transformer.layers = self._layers
        self.anchor = nn.Parameter(torch.zeros(1))
        self.temp = 0.3
        self.noise_clamp = None
        self.sampler_decode_steps = 1
        self.config = SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000))

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @property
    def sample_rate(self) -> int:
        return self.config.mimi.sample_rate

    def generate_audio_stream(self, _voice: object, _text: str) -> Iterator[torch.Tensor]:
        yield self._layers[5](torch.ones(1, 2, 4))


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "emotions.safetensors"
    save_file(
        {
            f"steering_vectors.{emotion}": torch.ones(4) * (index + 1)
            for index, emotion in enumerate(EMOTION_PROSODY_DEFAULTS)
        },
        path,
    )
    return path


def test_streaming_steering_keeps_original_model_unchanged(weights: Path) -> None:
    base = FakeTTS()
    wrapped = EmoTTS(cast(TTSModel, base), weights)
    neutral = next(wrapped.generate_audio_stream(None, "hello"))
    wrapped.set_emotion("happy", 0.5)
    shifted = next(wrapped.generate_audio_stream(None, "hello"))
    original = next(base.generate_audio_stream(None, "hello"))
    assert torch.equal(neutral, original)
    assert torch.equal(shifted, original + 2)
    assert base.temp == 0.3
    assert base.sampler_decode_steps == 1
    assert wrapped.model.sampler_decode_steps == 2
    assert wrapped.sample_rate == 24000
    wrapped.cleanup()
    assert wrapped._hook_handle is None
    assert torch.equal(next(wrapped.generate_audio_stream(None, "hello")), original)
    wrapped.cleanup()


def test_validation_and_layer_bounds_are_offline(weights: Path) -> None:
    with pytest.raises(TypeError):
        _profile(cast(dict[str, dict[str, object]], []))
    with pytest.raises(ValueError):
        _profile({"happy": {"temp": 1, "noise_clamp": 2, "lsd_decode_steps": True}})
    with pytest.raises(ValueError):
        EmoTTS(cast(TTSModel, FakeTTS()), weights, injection_layer=6)
    with pytest.raises(ValueError):
        EmoTTS(cast(TTSModel, FakeTTS()), "hf://someone/repo/weights.pt")


def test_safe_weights_only_loads_tensor_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save({"unexpected": torch.zeros(4)}, path)
    with pytest.raises(ValueError, match="six emotion vectors"):
        EmoTTS(cast(TTSModel, FakeTTS()), path)
