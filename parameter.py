# lora 参数
SF = 9
BW = 125e3
FS = 1e6
CHIRP_LEN = round(FS/BW*pow(2,SF))
SNR = 21

N_FFT = 64 # stft所有长度
N_DATA = 1000  # data总数 

# 发送端参数
# n_sender 发送端数量，mix_num每次随机用多少个发送端混合
N_SENDER = 10
N_MIXER = 3
TOTAL_LEN = CHIRP_LEN * N_MIXER

# model 参数
SENDER_EMB = 10
DELAY_EMB = 10