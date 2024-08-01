import torch
import torchaudio
from agc import AGC
import julius
import torch.nn.functional as F

device = 0
agc = AGC.from_pretrained("Audiogen/agc-discrete").to(0)

def convert_audio(wav: torch.Tensor, from_rate: float) -> torch.Tensor:
    wav = julius.resample_frac(wav, int(from_rate), 48000)
    return wav


def load_audio():
    audio1, sr = torchaudio.load("D:\Dictionary Learning Dataset\Oliver Power Tools Sample Pack II\OLIVER_sample_pack\OLIVER_tonal\OLIVER_arpeggios\OLIVER_arpeggio_loop_reso_90_Fmaj.wav")
    audio1 = convert_audio(audio1, sr)
    torchaudio.save("audio1.wav", audio1, 48000)
    padding = (0, 480000 - audio1.size(-1))
    audio1 = F.pad(audio1, padding, "constant", 0).to(0)

    audio2, sr = torchaudio.load("D:\Dictionary Learning Dataset\Oliver Power Tools Sample Pack II\OLIVER_sample_pack\OLIVER_tonal\OLIVER_arpeggios\OLIVER_arpeggio_loop_chemical_85_D#min.wav")
    audio2 = convert_audio(audio2, sr)
    torchaudio.save("audio2.wav", audio2, 48000)
    padding = (0, 480000 - audio2.size(-1))
    audio2 = F.pad(audio2, padding, "constant", 0).to(0)
    return audio1, audio2


audio1, audio2 = load_audio()

with torch.no_grad():
    z1 = agc.encode(audio1[None, :, :])
    z2 = agc.encode(audio2[None, :, :])
    # 1, 24, 500

    reconstructed_audio = agc.decode(z1).cpu()
    torchaudio.save("audio1.wav", reconstructed_audio[0], 48000)
    reconstructed_audio = agc.decode(z2).cpu()
    torchaudio.save("audio2.wav", reconstructed_audio[0], 48000)

    # max_val = z.max()
    #
    # perturbations = 6000
    #
    # for i in range(perturbations):
    #     index = (0, torch.randint(0, 24, (1,)).item(), torch.randint(0, 500, (1,)).item())
    #     z[index] = min(max_val, z[index] + 10)

    alpha = .99999998
    z3 = (z1 * (1 - alpha) + z2 * alpha).to(torch.int)

    reconstructed_audio = agc.decode(z3).cpu()
    torchaudio.save("result.wav", reconstructed_audio[0], 48000)
