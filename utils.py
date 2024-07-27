import julius
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import einops
from torch.utils.data.dataloader import default_collate


def convert_audio(wav: torch.Tensor, from_rate: float) -> torch.Tensor:
    if wav.shape[0] == 1:
        wav = einops.repeat(wav, '1 w -> 2 w')
    wav = julius.resample_frac(wav, int(from_rate), 48000)
    return wav


class AudioDataset(Dataset):
    def __init__(self, ):
        self.directory = "D:\\Dictionary Learning Dataset"
        self.audio_files = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith('.wav'):
                    self.audio_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"File loaded")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
        waveform = convert_audio(waveform, sample_rate)

        padding = (0, 480000 - waveform.size(-1))
        waveform = F.pad(waveform, padding, "constant", 0)
        return waveform


def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def get_dataloader(bs):
    dataset = AudioDataset()
    loader = DataLoader(dataset, batch_size=bs, collate_fn=safe_collate, shuffle=True, num_workers=6)
    return loader
