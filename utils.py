import julius
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import einops
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def convert_audio(wav: torch.Tensor, from_rate: float) -> torch.Tensor:
    if wav.shape[0] == 1:
        wav = einops.repeat(wav, '1 w -> 2 w')
    wav = julius.resample_frac(wav, int(from_rate), 48000)
    return wav


class AudioDataset(Dataset):
    def __init__(self, ):
        self.directory = "E:\\Dictionary Learning Dataset"
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
            # print(f"File loaded")
        except Exception as e:
            # print(f"Error loading file: {e}")
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
    # sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=bs, collate_fn=safe_collate, shuffle=False)
    return loader


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * .1)
        self.decay_start_step = int(total_steps * .9)
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        elif step < self.decay_start_step:
            progress = (step - self.warmup_steps) / (self.decay_start_step - self.warmup_steps)
            return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]
        else:
            progress = (step - self.decay_start_step) / (self.total_steps - self.decay_start_step)
            return [base_lr * (1 - progress) for base_lr in self.base_lrs]
