import torch
import math
import numpy as np
import cmath
from parameter import *
import torch.nn.functional as F

## ---------------------- LoRa ------------------------- ##
# add noise
def awgn(signal):
    # 计算信号功率
    P = torch.sum(torch.abs(signal)**2) / len(signal)

    # 计算噪声功率，以生成SNR为21dB
    N = P / (10**(SNR / 10.0))

    # 生成噪声
    real_noise = torch.randn_like(signal.real) * torch.sqrt(N / 2)
    imag_noise = torch.randn_like(signal.imag) * torch.sqrt(N / 2)
    complex_noise = torch.complex(real_noise, imag_noise)

    # 将噪声添加到信号中
    return signal + complex_noise

# generate chirp
def chirp(sf, bw, fs, begin_pos=0):
    # begin_pos in [0, 2^sf)
    assert type(begin_pos) == int
    assert begin_pos >= 0 and begin_pos < pow(2,sf)

    N = pow(2, sf)
    T = N/bw
    samp_per_sym = round(fs/bw*N)

    k = bw/T
    f0 = -bw/2

    t = np.arange(samp_per_sym*(N-begin_pos)/N + 1)/fs
    snum = t.shape[0]
    c1 = np.exp(1j * 2*np.pi * (t * (f0 + k*T*begin_pos/N + 0.5*k*t)))

    if snum == 0:
        phi = 0.
    else:
        phi = cmath.phase(c1[snum-1])

    t = np.arange(samp_per_sym*begin_pos/N)/fs
    c2 = np.exp(1j * (phi + 2*np.pi*(t * (f0 + 0.5 *k*t))))

    return np.concatenate((c1[:snum-1], c2))

# demodulate the chirp
def dechirp(chirp):
    downchirp = torch.from_numpy(np.conjugate(chirp(SF, BW, FS)))

    dechirp = torch.mul(downchirp, chirp)

    fftres = torch.fft(dechirp)
    fftabs = torch.abs(fftres)

    max_index = torch.argmax(fftabs)

    if max_index >= pow(2, SF-1):
        max_index = pow(2,SF) - (round(FS/BW*pow(2,SF)) - max_index)

    return max_index


# resample
def upsample(signal, factor):
    # signal: [batch_size, CHIRP_LEN]
    _, chirp_len = signal.shape
    new_length = chirp_len * factor

    interpolated_real = F.interpolate(signal.real.unsqueeze(1), size=new_length, mode='linear')
    print(interpolated_real.shape)
    interpolated_imag = F.interpolate(signal.imag.unsqueeze(1), size=new_length, mode='linear')
    # 合并为插值后的复数信号
    interpolated_complex_signal = torch.view_as_complex(torch.stack([interpolated_real, interpolated_imag], dim=-1))
    return interpolated_complex_signal.squeeze(1)


## ---------------------- DL --------------------------- ##
def positional_embedder(pos, d_embed=100):
    # pos: [batch_size, N ,1], d_embed: 编码特征维度
    batch_size, n_sender, _ = pos.shape
    # PE(pos,2i) = sin( pos / 10000^(2i/d) )
    div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed)).cuda()     # d_embed/2
    div_term = div_term[None, None, :].repeat([batch_size, n_sender, 1])
    pos_enc = torch.zeros(batch_size, n_sender, d_embed).cuda()
    pos_enc[:, :, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, :, 1::2] = torch.cos(pos * div_term)
    return pos_enc

# mix the signal in time domain
def mix(encoded_signals, delays):
    # encoded_signals: [batch_size, N_MIXER, chirp_len]
    # delays: [batch_size, N_MIXER, 1]
    batch_size, _, _ = encoded_signals.shape

    # mixed_signal: [batch_size, N_MIXER, total_len]
    mixed_signal = torch.zeros(batch_size, TOTAL_LEN, dtype=torch.complex64, requires_grad=True).cuda()
    for b in range(batch_size):
        for i in range(N_MIXER):
            offset = delays[b][i]
            mixed_signal[b][offset : offset+CHIRP_LEN] += encoded_signals[b][i]

    return awgn(mixed_signal)

# decode the signal
def decode(mixed_signal, encoded_signals, delays):
    # mixed_signal: [batch_size, N_MIXER, chirp_len]
    # encoded_signals: [batch_size, N_MIXER, chirp_len]
    # delays: [batch_size, N_MIXER, 1]    
    batch_size, _ = mixed_signal.shape
    
    dechirps = torch.zeros(batch_size, N_MIXER, CHIRP_LEN, dtype=torch.complex64, requires_grad=True).cuda()

    for b in range(batch_size):
        for i in range(N_MIXER):   
            offset = delays[b][i] 
            # plt.specgram(time_signals[b][i].cpu().detach().numpy(), Fs=FS)   
            # plt.show()

            dechirps[b][i] = torch.mul(torch.conj(encoded_signals[b][i]), mixed_signal[b][offset : offset+CHIRP_LEN])
            dechirps[b][i] = torch.fft.fft(dechirps[b][i])
        
        # plt.plot(abs(dechirps[b][i]).cpu().detach().numpy())
        # plt.show()

    return torch.abs(dechirps)

def cal_predict(encoding, delays, total_len):
    # encoding: [batch_size, n, chirp_len] 编码结果
    # delays : [batch_size, n, 1] 时延
    # symbols: [batch_size, n, 1] 真实symbol结果
    batch_size, n_mixer, chirp_len = encoding.shape

    mixed_signal = mix(encoding, delays, total_len)
    predict_symbols = torch.zeros(batch_size, n_mixer, 1).cuda()
    
    for b in range(batch_size):
        for i in range(n_mixer):
            offset = delays[b][i]
            predict_symbols[b][i] = dechirp(mixed_signal[b][i][offset : offset+chirp_len])
    
    
    return predict_symbols

def complex_l1_loss(input, target):
    return F.l1_loss(input.real, target.real) + F.l1_loss(input.imag, target.imag)
