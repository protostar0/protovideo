

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)

text = """Everyone wants the perfect strategy...
But they forget the mind behind it."

"You can master charts, indicators, and news.
But if you can't master your emotions?
You’ll lose to fear, greed, and doubt."

"Successful traders aren’t just skilled...
They're disciplined. Calm. Focused."

"They don’t chase. They don’t panic.
They stick to the plan — no matter what."

"Because in the end...
It's not the market you have to beat.
It’s yourself."

"Master your mindset —
and the profits will follow."""
# If you want to synthesize with a different voice, specify the audio prompt
# AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(
    text, 
    # audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=5.0,
    cfg_weight=0.5
    )
ta.save("test-4.wav", wav, model.sr)