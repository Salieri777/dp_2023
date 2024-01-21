# import pydevd_pycharm
# pydevd_pycharm.settrace('166.111.81.124', port=33777, stdoutToServer=True, stderrToServer=True)

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from models import *
from torch.optim import Adam
from tqdm import tqdm
import pickle


from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import random
from lora_data import *
from parameter import *
from Utils import *

writer = SummaryWriter(log_dir='logs')
batch_size = 1
epochs = 3
dataset = LoRaDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 定义 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

curving_dataset = Curving_LoRaDataset()
_, curving_dataset = random_split(curving_dataset, [train_size, test_size])
curving_loader = DataLoader(curving_dataset, batch_size=batch_size, shuffle=False)

scatter_dataset = Scatter_LoRaDataset()
_, scatter_dataset = random_split(scatter_dataset, [train_size, test_size])
scatter_loader = DataLoader(scatter_dataset, batch_size=batch_size, shuffle=False)

combine_dataset = Combine_LoRaDataset()
_, combine_dataset = random_split(combine_dataset, [train_size, test_size])
combine_loader = DataLoader(combine_dataset, batch_size=batch_size, shuffle=False)

model = LoRaModel().cuda()
optimizer = Adam(model.parameters(), lr=6e-4)
scheduler = ExponentialLR(optimizer, gamma=0.999)
scheduler = MultiStepLR(optimizer, milestones=[300, 500, 700], gamma=0.5)


def train():
    model.train()
    loss_all=[]
    pbar = tqdm(total=epochs)
    it = 0

    for epoch in range(epochs+1):
        for signals, symbols, delays, senders, amps, probs in train_loader:
            # signals: [batch_size, N_MIXER, CHIRP_LEN]
            # symbols: [_, _, 1]
            # delays: [_, _, 1]
            # senders: [_, _, 1]
            # amps: [_, _, 1]
            # probs: [_, _, 1]

            signals = signals.view(batch_size*N_MIXER, CHIRP_LEN)
            signals = torch.stft(signals, n_fft=N_FFT, return_complex=True).cuda()
            signals = signals.view(batch_size, N_MIXER, N_FFT, -1)
            # signals: [_, _, N_fft, N_frame] 每个人的短时傅里叶变化结果
            symbols = symbols.cuda()
            delays = delays.cuda()
            amps = amps.cuda()
            delays_embed = positional_embedder(delays, d_embed=DELAY_EMB)
            senders = senders.cuda()
            probs = probs.cuda()

            encoded_signals = model(symbols, signals, delays_embed, senders, amps, probs)
            # encoded_signals: [_, _, CHIRP_LEN]

            mixed_signal = mix(symbols, encoded_signals, delays, amps)

            decode_res = decode(mixed_signal, encoded_signals, delays, amps)

            decode_res = decode_res.view(-1, decode_res.shape[-1])
            symbols = symbols.view(-1)

            loss = F.cross_entropy(decode_res, symbols).cuda()

            pbar.set_description(f"loss:{loss.item():.4f}")
            loss_all.append(loss.item())
            # loss_all = np.hstack((loss_all,loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/loss', loss.item(), it)
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], it)
            it += 1

            # eval
            if it % 5 == 0:
                with torch.no_grad():
                    signals, symbols, delays, senders, amps, probs = next(iter(test_loader))
                    # signals: [batch_size, N, 100], delays: [batch_size, N, 1], mask: [batch_size, N, 100]
                    signals = signals.view(batch_size*N_MIXER, CHIRP_LEN)
                    signals = torch.stft(signals, n_fft=N_FFT, return_complex=True).cuda()
                    signals = signals.view(batch_size, N_MIXER, N_FFT, -1)
                    symbols = symbols.cuda()
                    delays = delays.cuda()
                    amps = amps.cuda()
                    probs = probs.cuda()
                    delays_embed = positional_embedder(delays, d_embed=DELAY_EMB)
                    senders = senders.cuda()
                    encoded_signals = model(symbols, signals, delays_embed, senders, amps, probs)
                    # encoded_signals: [atch_size, N_MIXER, CHIRP_LEN]
                    mixed_signal = mix(symbols, encoded_signals, delays, amps)
                    decode_res = decode(mixed_signal, encoded_signals, delays, amps)
                    decode_res = decode_res.view(-1, decode_res.shape[-1])
                    symbols = symbols.view(-1)
                    _, decode_res = torch.max(decode_res, -1)

                    # 按位比较是否相等
                    elementwise_equal = torch.eq(decode_res, symbols)
                    # 计算相等的概率
                    BER = torch.mean(elementwise_equal.float())
                    print(f"BER:{BER.item():.4f}")

        pbar.update(1)
        scheduler.step()

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/ckpt_last.pth')
    plt.figure()
    plt.plot(np.arange(len(loss_all)), loss_all)
    plt.title('Unet Train Loss Decay')
    # plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.savefig('./loss_image/train_loss')

    with open('loss.pkl', 'wb') as file:
        pickle.dump(loss_all, file)

def evaluate():
    model.eval()

    total_BER = []

    pbar = tqdm(total=epochs)
    for it in range(epochs+1):
        for signals, symbols, delays, senders, amps, probs in test_loader:

            signals = signals.view(batch_size*N_MIXER, CHIRP_LEN)
            signals = torch.stft(signals, n_fft=N_FFT, return_complex=True).cuda()
            signals = signals.view(batch_size, N_MIXER, N_FFT, -1)
            symbols = symbols.cuda()
            delays = delays.cuda()
            amps = amps.cuda()
            probs = probs.cuda()
            delays_embed = positional_embedder(delays, d_embed=DELAY_EMB)
            senders = senders.cuda()

            encoded_signals = model(symbols, signals, delays_embed, senders, amps, probs)
            # encoded_signals: [batch_size, N_MIXER, CHIRP_LEN]

            mixed_signal = mix(symbols, encoded_signals, delays, amps)

            decode_res = decode(mixed_signal, encoded_signals, delays, amps)

            decode_res = decode_res.view(-1, decode_res.shape[-1])
            symbols = symbols.view(-1)
            _, decode_res = torch.max(decode_res, -1)
            # 按位比较是否相等
            elementwise_equal = torch.eq(decode_res, symbols)
            # 计算相等的概率
            BER = torch.mean(elementwise_equal.float())

            pbar.set_description(f"Evaluate/BER:{BER.item():.4f}")
            total_BER.append(BER.item())

            
        pbar.update(1)
        #writer.add_scalar('Evaluate/BER', BER.item(), it)
        print('Final Evaluate/BER', sum(total_BER) / len(total_BER))

def baseline(loader):
    with torch.no_grad():
        total_BER = []

        for it in range(epochs+1):
            for signals, symbols, delays, senders, amps, probs in loader:
                # signals: [batch_size, N_MIXER, CHIRP_LEN]

                signals = signals.cuda()
                symbols = symbols.cuda()
                delays = delays.cuda()
                amps = amps.cuda()

                mixed_signal = mix(symbols, signals, delays, amps)

                decode_res = decode(mixed_signal, signals, delays, amps)

                decode_res = decode_res.view(-1, decode_res.shape[-1])
                symbols = symbols.view(-1)
                _, decode_res = torch.max(decode_res, -1)
                # 按位比较是否相等
                elementwise_equal = torch.eq(decode_res, symbols)
                # 计算相等的概率
                BER = torch.mean(elementwise_equal.float())

                #pbar.set_description(f"Evaluate/BER:{BER.item():.4f}")
                total_BER.append(BER.item())


            #writer.add_scalar('Evaluate/BER', BER.item(), it)
        
        print('Final Evaluate/BER', sum(total_BER) / len(total_BER))

if __name__ == '__main__':
    seed = 0              # 统一固定seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.autograd.set_detect_anomaly(True)

    train()

    # 加载保存的模型权重
    # model.load_state_dict(torch.load('checkpoints/ckpt_last.pth'))

    evaluate()

    # print("lora")
    # baseline(test_loader)

    # print("curving")
    # baseline(curving_loader)

    # print("scatter")
    # baseline(scatter_loader)

    # print("combine")
    # baseline(combine_loader)
