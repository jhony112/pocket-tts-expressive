# Experimental emotion steering

This optional wrapper integrates the approach from upstream
[EmoTTS PR #194](https://github.com/kyutai-labs/pocket-tts/pull/194) with current
PocketTTS inference. The normal `TTSModel` training and streaming APIs are unchanged.
Emotion steering is experimental: the quality and transfer of third-party vectors
to a fine-tuned language model must be evaluated with listening tests.

Provide a trusted local steering-weight file (`.safetensors`, `.pt`, or `.pth`)
containing `steering_vectors.<emotion>` tensors for angry, disgust, fear, happy,
neutral, and sad. PyTorch files load with `weights_only=True`. A Hugging Face
`hf://` path is accepted only when it includes a pinned `@revision`.

```python
from pocket_tts.emotts import EmoTTS

with EmoTTS.load_model(weights_path="/path/to/emotions.safetensors") as model:
    voice = model.get_state_for_audio_prompt("alba")
    model.set_emotion("happy", intensity=0.5)
    for chunk in model.generate_audio_stream(voice, "Hello there."):
        play(chunk)
```

`EmoTTS` wraps a private model instance and delegates PocketTTS inference calls.
Passing an existing `TTSModel` copies it to keep the original unsteered. Calling
`EmoTTS.load_model` avoids this extra copy. `set_emotion(None)` restores the
model's original generation settings. Call `cleanup()` or use the context manager
when finished. PocketTTS models are not thread-safe; use separate instances for
concurrent streams.

This patch does not implement text tags, delivery styles, pace controls, or
training-time emotion conditioning.
