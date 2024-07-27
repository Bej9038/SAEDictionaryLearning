import julius
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os


def convert_audio(wav: torch.Tensor, from_rate: float) -> torch.Tensor:
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
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = convert_audio(waveform, sample_rate)
        return waveform


def get_dataloader(bs):
    dataset = AudioDataset()
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)
    return loader
