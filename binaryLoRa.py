import torch
from torch.utils.data import Dataset, DataLoader
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


def train():
    batch_size = 2
    dataset = LoRaDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = LoRaModel().cuda()
    optimizer = Adam(model.parameters(), lr=6e-4)       # 大于7e-4就不行
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    scheduler = MultiStepLR(optimizer, milestones=[300, 500, 700], gamma=0.5)

    pbar = tqdm(total=100)
    for it in range(1001):
        for signals, symbols, delays, senders in dataloader:
            # signals: [batch_size, N, 100], delays: [batch_size, N, 1], mask: [batch_size, N, 100]
            signals = signals.cuda()
            symbols = symbols.cuda()
            delays = delays.cuda()
            delays_embed = positional_embedder(delays, d_embed=DELAY_EMB)
            senders = senders.cuda()

            predict = model(signals, delays_embed, senders, delays)
        
            predict = predict.view(-1, predict.shape[-1])
            symbols = symbols.view(-1)

            loss = F.cross_entropy(predict, symbols)

            # for b in range(batch_size):
            #     for i in range():
            #     loss += F.cross_entropy(predict[b].unsqueeze(0), symbols[b].long())
            loss /= predict.shape[0]

            # predict: [batch_size, N_mixer, 1]

            # loss = F.l1_loss(predict[mask], signals.flatten())
            #loss = complex_l1_loss(predict[mask], signals.flatten())      # 实部虚部分别计算l1_loss比计算模长的l1_loss稍好
            pbar.set_description(f"loss:{loss.item():.4f}")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.update(1)
        scheduler.step()
        # print(predict[0][mask[0]].reshape(signals[0].shape)[0], signals[0][0])
        writer.add_scalar('Train/loss', loss.item(), it)
        # writer.add_scalar('Train/acc', accuracy.item(), it)
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], it)

        if it % 500 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/ckpt_{it:05d}.pth')
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/ckpt_last.pth')


if __name__ == '__main__':
    seed = 0              # 统一固定seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    train()