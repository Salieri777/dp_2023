import torch
from torch.utils.data import Dataset
import numpy as np
from parameter import *
from Utils import *
import matplotlib.pyplot as plt

class LoRaDataset(Dataset):
    def __init__(self):
        self.prob = np.random.randint(low=1, high=100, size=N_SENDER)
        self.prob = self.prob / np.sum(self.prob)

    def __len__(self):
        # 信号总长度
        return N_DATA

    def __getitem__(self, item):
        # 一个lora_data
        # signals [N_MIXER, N_fft, N_frame] 每个人的短时傅里叶变化结果 
        # delays [N_MIXER, 1] 每个人的时延
        signals = []
        symbols = []
        delays = []
        amps = []

        senders = np.random.choice(np.arange(N_SENDER), N_MIXER, replace=False, p=self.prob)
        senders = torch.LongTensor(senders)
        for _ in range(N_MIXER):
            symbols.append(torch.zeros(1, dtype=torch.long))

            signals.append(torch.tensor(chirp(SF, BW, FS), dtype=torch.complex64))

            delays.append(torch.randint(TOTAL_LEN - CHIRP_LEN, [1]))

            amps.append(torch.randint(low=500, high=N_AMP, size=[1]))

        signals = torch.stack(signals, 0)
        signals = torch.stft(signals, n_fft=N_FFT, return_complex=True)
        symbols = torch.stack(symbols, 0)
        delays = torch.stack(delays, 0)
        amps = torch.stack(amps, 0)
        p = self.prob

        return signals, symbols, delays, senders, amps, p
