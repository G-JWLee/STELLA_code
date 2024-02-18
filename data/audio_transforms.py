from torchvision import transforms as T
from data.transforms import audio as aT2


class Audio_transform:
    """
    Audio transform in CAV
    """
    def __init__(self,
                 audio_rate=16000,
                 num_mels=128,
                 audio_size=1024,
                 freqm=0,
                 timem=0,
                 mean=-5.081,
                 std=4.4849,
                 noise=False,
                 training=False,
                 ):

        self.audio_rate = audio_rate
        self.freqm = freqm
        self.timem = timem
        self.noise = noise and training

        if training:
            transforms = [
                aT2.ToTensor(),
                aT2.ToMono(),
                aT2.CAV_Wav2fbank(
                    sampling_rate=audio_rate,
                    num_mel_bins=num_mels,
                    target_length=audio_size,
                    freqm=freqm,
                    timem=timem,
                ),
                aT2.CAV_Normalize(mean, std),
            ]
        else:
            transforms = [
                aT2.ToTensor(),
                aT2.ToMono(),
                aT2.CAV_Wav2fbank(
                    sampling_rate=audio_rate,
                    num_mel_bins=num_mels,
                    target_length=audio_size,
                    freqm=0,
                    timem=0,
                ),
                aT2.CAV_Normalize(mean, std)
            ]

        self.t = T.Compose(transforms)
        self.noise_t = T.Compose([aT2.Noise(audio_size)])

    def __call__(self, x):
        spec = self.t(x)
        if self.noise:
            spec = self.noise_t(spec)
        return spec


class AVE_Audio_transform:
    """
    Audio transform in CAV
    """
    def __init__(self,
                 audio_rate=16000,
                 num_mels=128,
                 audio_size=1024,
                 freqm=0,
                 timem=0,
                 mean=-5.081,
                 std=4.4849,
                 noise=False,
                 training=False,
                 ):

        self.audio_rate = audio_rate
        self.freqm = freqm
        self.timem = timem
        self.noise = noise and training

        if training:
            transforms = [
                aT2.ToTensor(),
                aT2.ToMono(),
                aT2.AVE_Wav2fbank(
                    sampling_rate=audio_rate,
                    num_mel_bins=num_mels,
                    target_length=audio_size,
                    freqm=freqm,
                    timem=timem,
                ),
                aT2.CAV_Normalize(mean, std),
            ]
        else:
            transforms = [
                aT2.ToTensor(),
                aT2.ToMono(),
                aT2.AVE_Wav2fbank(
                    sampling_rate=audio_rate,
                    num_mel_bins=num_mels,
                    target_length=audio_size,
                    freqm=0,
                    timem=0,
                ),
                aT2.CAV_Normalize(mean, std)
            ]

        self.t = T.Compose(transforms)
        self.noise_t = T.Compose([aT2.Noise(audio_size)])

    def __call__(self, x):
        spec = self.t(x)
        if self.noise:
            spec = self.noise_t(spec)
        return spec