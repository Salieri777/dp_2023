import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from models import *
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import random
from lora_data import *
from parameter import *
from Utils import *

writer = SummaryWriter(log_dir='logs')
batch_size = 50
epoch = 10
dataset = LoRaDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 定义 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LoRaModel().cuda()
optimizer = Adam(model.parameters(), lr=6e-4)
scheduler = ExponentialLR(optimizer, gamma=0.999)
scheduler = MultiStepLR(optimizer, milestones=[300, 500, 700], gamma=0.5)


def train():
    model.train(True)

    pbar = tqdm(total=epoch)
    for it in range(epoch+1):
        for signals, symbols, delays, senders in train_loader:
            # signals: [batch_size, N, 100], delays: [batch_size, N, 1]
            signals = signals.cuda()
            symbols = symbols.cuda()
            delays = delays.cuda()
            delays_embed = positional_embedder(delays, d_embed=DELAY_EMB)
            senders = senders.cuda()

            encoded_signals = model(signals, delays_embed, senders)
            # encoded_signals: [batch_size, N_MIXER, CHIRP_LEN]

            mixed_signal = mix(encoded_signals, delays)

            decode_res = decode(mixed_signal, encoded_signals, delays)

            decode_res = decode_res.view(-1, decode_res.shape[-1])
            symbols = symbols.view(-1)

            loss = F.cross_entropy(decode_res, symbols)

            pbar.set_description(f"loss:{loss.item():.4f}")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.update(1)
        scheduler.step()
        writer.add_scalar('Train/loss', loss.item(), it)
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], it)

        if it % 500 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/ckpt_{it:05d}.pth')
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/ckpt_last.pth')

def evaluate():
    model.train(False)

    pbar = tqdm(total=epoch)
    for it in range(epoch+1):
        for signals, symbols, delays, senders in test_loader:
            # signals: [batch_size, N, 100], delays: [batch_size, N, 1], mask: [batch_size, N, 100]
            signals = signals.cuda()
            symbols = symbols.cuda()
            delays = delays.cuda()
            delays_embed = positional_embedder(delays, d_embed=DELAY_EMB)
            senders = senders.cuda()

            encoded_signals = model(signals, delays_embed, senders)
            # encoded_signals: [batch_size, N_MIXER, CHIRP_LEN]

            mixed_signal = mix(encoded_signals, delays)

            decode_res = decode(mixed_signal, encoded_signals, delays)

            decode_res = decode_res.view(-1, decode_res.shape[-1])
            symbols = symbols.view(-1)
            _, decode_res = torch.max(decode_res, -1)
            # 按位比较是否相等
            elementwise_equal = torch.eq(decode_res, symbols)
            # 计算相等的概率
            BER = torch.mean(elementwise_equal.float())

            pbar.set_description(f"Evaluate/BER:{BER.item():.4f}")


        pbar.update(1)
        writer.add_scalar('Evaluate/BER', BER.item(), it)


if __name__ == '__main__':
    seed = 0              # 统一固定seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.autograd.set_detect_anomaly(True)

    train()

    # evaluate()
