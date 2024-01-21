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
        self.amp = np.random.randint(A_LOW, A_HIGH, N_SENDER, dtype=int).tolist()
    
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

        # senders = np.random.choice(np.arange(N_SENDER), N_MIXER, replace=False, p=self.prob)
        senders = np.arange(N_MIXER)
        
        amps = [torch.tensor(self.amp[i]) for i in senders]
        
        senders = torch.LongTensor(senders)
        
        for _ in range(N_MIXER):
            # symbols.append(torch.zeros(1, dtype=torch.long))

            # signals.append(torch.tensor(chirp(SF, BW, FS), dtype=torch.complex64))

            symbol = torch.randint(low=0, high=pow(2,SF), size=[1])
            # symbol = torch.full([1], i)
            # symbol = torch.zeros(1, dtype=torch.long)
            symbols.append(symbol)
            signals.append(torch.tensor(chirp(SF, BW, FS), dtype=torch.complex64))

            delays.append(torch.randint(TOTAL_LEN - CHIRP_LEN, [1]))

        signals = torch.stack(signals, 0)
        symbols = torch.stack(symbols, 0)
        delays = torch.stack(delays, 0)
        amps = torch.stack(amps, 0)
        p = self.prob

        return signals, symbols, delays, senders, amps, p
    
class Curving_LoRaDataset(Dataset):
    def __init__(self):
        self.prob = np.random.randint(low=1, high=100, size=N_SENDER)
        self.prob = self.prob / np.sum(self.prob)
        self.amp = np.random.randint(A_LOW, A_HIGH, N_SENDER, dtype=int).tolist()
    

    def __len__(self):
        # 信号总长度
        return N_DATA

    def __getitem__(self, item):
        # 一个lora_data
        # signals [N_MIXER, CHIRP_LEN] 
        # delays [N_MIXER, 1]
        signals = []
        symbols = []
        delays = []

        # senders = np.random.choice(np.arange(N_SENDER), N_MIXER, replace=False, p=self.prob)
        senders = np.arange(N_MIXER)
        
        amps = [torch.tensor(self.amp[i]) for i in senders]
        
        senders = torch.LongTensor(senders)
        
        for _ in range(N_MIXER):
            symbol = torch.randint(low=0, high=pow(2, SF), size=[1])
            symbols.append(symbol)
            signals.append(torch.tensor(curving_chirp(SF, BW, FS), dtype=torch.complex64))

            delays.append(torch.randint(TOTAL_LEN - CHIRP_LEN, [1]))

        signals = torch.stack(signals, 0)
        symbols = torch.stack(symbols, 0)
        delays = torch.stack(delays, 0)
        amps = torch.stack(amps, 0)
        p = self.prob

        return signals, symbols, delays, senders, amps, p
    
class Scatter_LoRaDataset(Dataset):
    def __init__(self):
        self.prob = np.random.randint(low=1, high=100, size=N_SENDER)
        self.prob = self.prob / np.sum(self.prob)
        self.amp = np.random.randint(A_LOW, A_HIGH, N_SENDER, dtype=int).tolist()
    
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

        # senders_cpu = np.random.choice(np.arange(N_SENDER), N_MIXER, replace=False, p=self.prob)
        senders_cpu = np.arange(N_MIXER)
        
        amps = [torch.tensor(self.amp[i]) for i in senders_cpu]
        
        senders = torch.LongTensor(senders_cpu)
        
        for i in range(N_MIXER):
            symbol = torch.randint(low=0, high=pow(2, SF), size=[1])
            symbols.append(symbol)
            signals.append(torch.tensor(scatter_chirp(SF, BW, BW, m=senders_cpu[i]), dtype=torch.complex64))

            delays.append(torch.randint(TOTAL_LEN - CHIRP_LEN, [1]))

        signals = torch.stack(signals, 0)
        symbols = torch.stack(symbols, 0)
        delays = torch.stack(delays, 0)
        amps = torch.stack(amps, 0)
        p = self.prob

        return signals, symbols, delays, senders, amps, p

class Combine_LoRaDataset(Dataset):
    def __init__(self):
        self.prob = np.random.randint(low=1, high=100, size=N_SENDER)
        self.prob = self.prob / np.sum(self.prob)
        self.amp = np.random.randint(A_LOW, A_HIGH, N_SENDER, dtype=int).tolist()
    
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

        # senders_cpu = np.random.choice(np.arange(N_SENDER), N_MIXER, replace=False, p=self.prob)
        senders_cpu = np.arange(N_MIXER)
        
        amps = [torch.tensor(self.amp[i]) for i in senders_cpu]
        
        senders = torch.LongTensor(senders_cpu)
        
        for i in range(N_MIXER):
            symbol = torch.randint(low=0, high=pow(2, SF), size=[1])
            symbols.append(symbol)

            tmp1 = curving_chirp(SF, BW, BW)
            tmp2 = scatter(tmp1, m=senders_cpu[i])
            signals.append(torch.tensor(tmp2, dtype=torch.complex64))

            delays.append(torch.randint(TOTAL_LEN - CHIRP_LEN, [1]))

        signals = torch.stack(signals, 0)
        symbols = torch.stack(symbols, 0)
        delays = torch.stack(delays, 0)
        amps = torch.stack(amps, 0)
        p = self.prob

        return signals, symbols, delays, senders, amps, p
