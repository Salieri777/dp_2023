import torch
import math
import numpy as np
import cmath
from parameter import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

## ---------------------- LoRa ------------------------- ##
# add noise
def awgn(signal, snr=SNR):

    # plt.specgram(signal.cpu(), Fs=BW)
    # plt.show()

    # 计算信号功率
    # P = torch.sum(torch.abs(signal)**2, -1, keepdim=True) / 1
    # 振幅为(0.5, 1)，对应功率(0.25, 1)。先默认取最低
    p = pow((A_LOW+A_HIGH)/2/A_HIGH, 2)
    # 计算噪声功率，以生成SNR为21dB
    n = p / (10**(snr / 10.0))

    size = (signal.shape[0])

    # 生成噪声    
    real_noise = torch.FloatTensor(size).normal_(mean=0, std=math.sqrt(n/2))
    imag_noise = torch.FloatTensor(size).normal_(mean=0, std=math.sqrt(n/2))

    complex_noise = torch.complex(real_noise.cuda(), imag_noise.cuda())

    # tmp = signal + complex_noise
    # plt.specgram(tmp.cpu(), Fs=BW)
    # plt.show()

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

def curving_chirp(sf, bw, fs, begin_pos=0):
    # begin_pos in [0, 2^sf)
    assert type(begin_pos) == int
    assert begin_pos >= 0 and begin_pos < pow(2,sf)

    N = pow(2, sf)
    T = N/bw
    samp_per_sym = round(fs/bw*N)
    
    f0 = -bw/2

    t = np.arange(samp_per_sym) / fs
    signal = np.exp(1j * 2*np.pi * (f0 * t +  BW/(4*pow(T, 3)) *t*t*t*t))

    return signal

def single_tone(T, fs, f):
    samp_per_sym = round(T * fs)
    t = np.arange(samp_per_sym) / fs
    s = np.exp(1j * 2*np.pi * f * t)    
    return s


def split_and_rearrange_array(arr, m, order):
    if m <= 0 or len(order) != m:
        return "Invalid values for m or order"

    # 计算每段的长度
    segment_length = len(arr) // m
    remainder = len(arr) % m

    # 切片并重新排列
    rearranged_array = []


    for i in order:
        start_index = i*segment_length        
        end_index = start_index + segment_length + (remainder if i == m-1 else 0)
        segment = arr[start_index:end_index]
        rearranged_array.extend(segment)

    return rearranged_array

def scatter_chirp(sf, bw, fs, m, begin_pos=0, M=N_SENDER, b=7):
    assert m >=0 and m <= M-1
    
    normal_chirp = chirp(sf, bw, fs, begin_pos)
    s0 = list(range(M))
    sm = []
    for s in s0:
        sm.append((s * m + b) % M)

    return normal_chirp if m == 0 else split_and_rearrange_array(normal_chirp, M, sm)

def scatter(chirp, m, M=N_SENDER, b=7):
    assert m >=0 and m <= M-1
    
    normal_chirp = chirp
    s0 = list(range(M))
    sm = []
    for s in s0:
        sm.append((s * m + b) % M)

    return normal_chirp if m == 0 else split_and_rearrange_array(normal_chirp, M, sm)

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
    # signal: [batch_size, channel_size,siz1,size2]
    _, chirp_len = signal.shape
    new_length = chirp_len * factor

    interpolated_real = F.interpolate(signal.real.unsqueeze(1), size=new_length, mode='linear')
    print(interpolated_real.shape)
    interpolated_imag = F.interpolate(signal.imag.unsqueeze(1), size=new_length, mode='linear')
    # 合并为插值后的复数信号
    interpolated_complex_signal = torch.view_as_complex(torch.stack([interpolated_real, interpolated_imag], dim=-1))
    return interpolated_complex_signal.squeeze(1)

def upsample4(signal, newsize3,newsize4):
    # signal: [batch_size, channel_size,siz3,size4]
    
    interpolated_real = F.interpolate(signal.real, size=(newsize3,newsize4), mode='bilinear')
    # print(interpolated_real.shape)
    interpolated_imag = F.interpolate(signal.imag, size=(newsize3,newsize4), mode='bilinear')
    # 合并为插值后的复数信号
    interpolated_complex_signal = torch.view_as_complex(torch.stack([interpolated_real, interpolated_imag], dim=-1)).cuda()
    return interpolated_complex_signal


def max_pool_2d(signal, kernel_size):

    # 假设input_tensor的shape为(batch_size, channels, height, width)
    batch_size, channels, height, width = signal.shape
    height=(height // kernel_size)*kernel_size
    width=(width // kernel_size)*kernel_size
    input_tensor = signal[:,:,0:height,0:width].abs()
    
    print('input_tensor',input_tensor.shape)

    # 设定池化窗口大小
    input_tensor = torch.randn(batch_size, channels, height, width)

    # 将通道和空间维度合并为一个维度
    merged_dimensions = input_tensor.view(batch_size, channels, height // kernel_size, kernel_size, width // kernel_size, kernel_size)
    print('merged_dimensions',merged_dimensions.shape)

    # 在合并的维度上进行最大化
    max_values = torch.max(merged_dimensions, dim=3)
    max_values = torch.max(max_values, dim=5)
    print('max_values',max_values.shape)
    

    # 将结果重塑回原始形状
    output_tensor = max_values.view(batch_size, channels, height // kernel_size, width // kernel_size)
    return output_tensor
    

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
def mix(symbols, signals, delays, amps):
    # symbols: [_, _, 1]
    # signals: [batch_size, N_MIXER, CHIRP_LEN]
    # delays: [batch_size, N_MIXER, 1]
    # amps: [batch_size, N_MIXER, 1] 500-1000
    batch_size, _, _ = signals.shape

    # plt.specgram(signals[0][0].cpu(), Fs=BW)   
    # plt.show()

    # mixed_signal: [batch_size, N_MIXER, total_len]
    mixed_signal = torch.zeros(batch_size, TOTAL_LEN, dtype=torch.complex64, requires_grad=True).cuda()
    for b in range(batch_size):
        for i in range(N_MIXER):
            offset = delays[b][i]
            
            symbol = symbols[b][i].item()
            t = pow(2, SF) / BW
            s = single_tone(t, BW, BW/pow(2, SF) * symbol)
            s = torch.tensor(s).cuda()         

            mixed_signal[b][offset : offset+CHIRP_LEN] += awgn(signals[b][i] * s * amps[b][i]/A_HIGH)

    return mixed_signal

# ------------------- To Do ---------------------------

# decode the signal
def decode(mixed_signal, signals, delays, amps):
    # mixed_signal: [batch_size, N_MIXER, chirp_len]
    # encoded_signals: [batch_size, N_MIXER, chirp_len]
    # delays: [batch_size, N_MIXER, 1]
    # amps: [batch_size, N_MIXER, 1]
    batch_size, _ = mixed_signal.shape
    
    dechirps = torch.zeros(batch_size, N_MIXER, CHIRP_LEN, dtype=torch.complex64, requires_grad=True).cuda()

    for b in range(batch_size):
        for i in range(N_MIXER):   
            offset = delays[b][i] 
            
            # plt.specgram(mixed_signal[b][offset : offset+CHIRP_LEN].cpu(), Fs=BW)   
            # plt.show()

            dechirps[b][i] = torch.mul(torch.conj(signals[b][i]), mixed_signal[b][offset : offset+CHIRP_LEN])
            dechirps[b][i] = torch.fft.fft(dechirps[b][i])
        
            # plt.figure()
            # plt.plot(abs(dechirps[b][i]).cpu())

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
