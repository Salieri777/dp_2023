import torch
from torch import nn

from parameter import *
from Utils import *
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexMaxPool2d, ComplexConvTranspose2d, ComplexReLU,ComplexAvgPool2d,ComplexMaxPool2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_matmul
import torch.nn.functional as F
import matplotlib.pyplot as plt

def complex_sigmoid(input):
    return F.sigmoid(input.real).type(torch.complex64)+1j*F.sigmoid(input.imag).type(torch.complex64)




# -------------------- To Do --------------------------
# -------------------- 对二维输入进行encoder -----------
# input (signals[batch_size, n_sender, n_fft, n_frame], delay_embed[~,~,DELAY_EMB], sender_embedding[~,~,SENDER_EMB]
# output (encoding[batch_size, n_sender, n_fft, n_frame])

class UNetDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.up = ComplexConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # up-conv 2*2
        self.conv_relu = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(),
            ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU()
        )

    def forward(self, high, low):
        x1 = self.up(high)
        offset1 = x1.size()[2] - low.size()[2]
        offset2 = x1.size()[3] - low.size()[3]
        padding = [offset2 // 2, offset2 // 2, offset1 // 2, offset1 // 2, ]
        # 计算应该填充多少（这里可以是负数）
        x2 = F.pad(low, padding)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class UNet_v1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ComplexConv2d(41, 64, 3),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexConv2d(64, 64, 3),
            ComplexBatchNorm2d(64),
            ComplexReLU()
        )
        self.layer2 = nn.Sequential(
            ComplexConv2d(64, 128, 3),
            ComplexBatchNorm2d(128),
            ComplexReLU(),
            ComplexConv2d(128, 128, 3),
            ComplexBatchNorm2d(128),
            ComplexReLU()
        )
        self.layer3 = nn.Sequential(
            ComplexConv2d(128, 256, 3),
            ComplexBatchNorm2d(256),
            ComplexReLU(),
            ComplexConv2d(256, 256, 3),
            ComplexBatchNorm2d(256),
            ComplexReLU()
        )
        self.layer4 = nn.Sequential(
            ComplexConv2d(256, 512, 3),
            ComplexBatchNorm2d(512),
            ComplexReLU(),
            ComplexConv2d(512, 512, 3),
            ComplexBatchNorm2d(512),
            ComplexReLU()
        )

        self.maxpool = ComplexMaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.decoder4 = UNetDecoder(1024, 512)
        self.decoder3 = UNetDecoder(512, 256)
        self.decoder2 = UNetDecoder(256, 128)
        self.decoder1 = UNetDecoder(128, 64)

        self.last = ComplexConv2d(64, 1, 1)

    def forward(self, input):
        # input: [batch_size*mix_num, 1+10+10, n_fft, n_frame]      [1signal + 10delay_embed + 10sender_embed]
        # output: [batch_size*mix_num, 1+10+10, n_fft, n_frame]

        # [21,1024,129]->[64,1022,127]->[64,1020,125]
        layer1 = self.layer1(input)
        # [64,1020,125]-pad->[64,1020,126]->[64,510,63]->[128,508,61]->[128,506,59]
        layer1_padded = torch.cat([layer1, layer1[...,-1:]],-1)
        layer2 = self.layer2(self.maxpool(layer1_padded))
        # [128,506,59]-pad->[128,506,60]->[128,253,30]->[256,251,28]->[256,249,26]
        layer2_padded = torch.cat([layer2, layer2[..., -1:]], -1)
        layer3 = self.layer3(self.maxpool(layer2_padded))
        # [256,249,26]-pad->[256,250,26]->[256,125,13]->[512,123,11]->[512,121,9]
        layer3_padded = torch.cat([layer3, layer3[:,:,-1:,:]], 2)
        layer4 = self.layer4(self.maxpool(layer3_padded))

        # Decoder
        layer5 = self.decoder3(layer4, layer3_padded)
        layer6 = self.decoder2(layer5, layer2_padded)
        layer7 = self.decoder1(layer6, layer1_padded)
        out = self.last(layer7)

        return out


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ComplexConv2d(1+DELAY_EMB+SENDER_EMB+AMP_EMB+PROB_EMB+SYMBOL_EMB, 32, kernel_size=5,stride=1,padding = 0),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            ComplexConv2d(32, 32, kernel_size=5,stride=1,padding = 0),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            # ComplexMaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.layer3 = nn.Sequential(
            ComplexConv2d(32, 64, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexConv2d(64, 64, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            # ComplexAvgPool2d(kernel_size=2)
        )
        self.layer5 = nn.Sequential(
            ComplexConv2d(64, 128, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(128),
            ComplexReLU(),
            ComplexConv2d(128, 128, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(128),
            ComplexReLU()
        )
        self.layer7 = nn.Sequential(
            ComplexConv2d(128, 256, kernel_size=1,stride=1,padding = 0),
            ComplexBatchNorm2d(256),
            ComplexReLU(),
            ComplexConv2d(256, 256, kernel_size=1,stride=1,padding = 0),
            ComplexBatchNorm2d(256),
            ComplexReLU()
        )
        self.layer8 = nn.Sequential(
            ComplexConv2d(256, 128, kernel_size=1,stride=1,padding = 0),
            ComplexBatchNorm2d(128),
            ComplexReLU(),
            ComplexConv2d(128, 128, kernel_size=1,stride=1,padding = 0),
            ComplexBatchNorm2d(128),
            ComplexReLU()
        )
        self.layer6 = nn.Sequential(
            ComplexConv2d(256, 64, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexConv2d(64, 64, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(64),
            ComplexReLU()
            # ComplexConvTranspose2d(64,64,2)

        )
        self.layer4 = nn.Sequential(
            ComplexConv2d(128, 32, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            ComplexConv2d(32, 32, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            # ComplexConvTranspose2d(32,32,2)
        )
        self.layer2 = nn.Sequential(
            ComplexConv2d(64, 1, kernel_size=5,stride=1,padding = 0),
            ComplexBatchNorm2d(1),
            ComplexReLU(),
            ComplexConv2d(1, 1, kernel_size=5,stride=1,padding = 0),
            ComplexBatchNorm2d(1),
            ComplexReLU()
            
        )

    def forward(self, x):
        # input: [batch_size*mix_num, 1+10+10, n_fft, n_frame]      [1signal + 10delay_embed + 10sender_embed]
        # output: [batch_size*mix_num, 1+10+10, n_fft, n_frame]

        # [N, 21,1024,129]->[N, 64,1024,129]
        
        layer1 = self.layer1(x)
        
        # layer1p = max_pool_2d(layer1,2)
        
        layer3 = self.layer3(layer1)
        
        layer5 = self.layer5(layer3)
        layer7 = self.layer7(layer5)
        layer8 = self.layer8(layer7)
        layer8_resize=upsample4(layer8, layer5.shape[2],layer5.shape[3])
        layer6 = self.layer6(torch.cat([layer8_resize, layer5], dim=1))
        layer6_resize=upsample4(layer6, layer3.shape[2],layer3.shape[3])
        layer4 = self.layer4(torch.cat([layer6_resize, layer3], dim=1))
        layer4_resize=upsample4(layer4, layer1.shape[2],layer1.shape[3])
        y = self.layer2(torch.cat([layer4_resize, layer1], dim=1))
        
        y_resize=upsample4(y, x.shape[2],x.shape[3])


        # layer1 = self.layer1(x)
        # [N,64,1024,129]->[N,1,1024,129]
        # x = self.layer4(layer1)
        return y_resize


class LoRaModel(torch.nn.Module):
    def __init__(self):
        super(LoRaModel, self).__init__()

        self.symbol_embeddings = nn.Parameter(torch.randn(pow(2, SF), SYMBOL_EMB, dtype=torch.complex64))
        self.sender_embeddings = nn.Parameter(torch.randn(N_SENDER, SENDER_EMB, dtype=torch.complex64))
        self.amp_embeddings = nn.Parameter(torch.randn(A_HIGH, AMP_EMB, dtype=torch.complex64))
        self.prob_embeddings = nn.Parameter(torch.randn(100, PROB_EMB, dtype=torch.complex64))
        self.encoder = UNet()
    
    
    def forward(self, symbols, signals, delay_embed, sender_id, amps, probs):
        # symbols: [batch_size, N_MIXER, 1]
        # signal: [batch_size, N_MIXER, N_FFT, N_FRAME]
        # delay_embed: [batch_size, N_MIXER, DELAY_EMB]
        # sender_id: [batch_size, N_MIXER]
        # delays: [batch_size, N_MIXER]
        # amps: [batch_size, N_MIXER ,1]
        # probs: [1, N_SENDER]
        batch_size, _, _, n_frame = signals.shape
        signals = signals.unsqueeze(2)      # [batch_size, N_MIXER, 1, N_FFT, N_FRAME]
        amps = amps.view(batch_size, N_MIXER)
        prob_embed = (torch.round(probs * 100)).long()
        prob_embed = prob_embed[0][sender_id]
        prob_embedding = self.prob_embeddings[prob_embed]
        sender_embedding = self.sender_embeddings[sender_id]

        sender_embedding = sender_embedding[..., None, None].repeat([1,1,1,N_FFT, n_frame])    # [batch_size, N_MIXER, 10, N_FFT, N_FRAME]
        delay_embed = delay_embed[..., None, None].repeat([1,1,1,N_FFT,n_frame])              # [batch_size, N_MIXER, 10, N_FFT, N_FRAME]
        prob_embedding = prob_embedding[..., None, None].repeat([1,1,1,N_FFT,n_frame])      # [batch_size, N_MIXER, 10, N_FFT, N_FRAME]
        amp_embed = self.amp_embeddings[amps]
        amp_embed = amp_embed[..., None, None].repeat([1,1,1,N_FFT,n_frame])    # [batch_size, N_MIXER, 10, N_FFT, N_FRAME]

        symbols = symbols.view(batch_size, N_MIXER)
        symbol_embed = self.symbol_embeddings[symbols]
        symbol_embed = symbol_embed[..., None, None].repeat([1,1,1,N_FFT,n_frame])

        # encode the input
        input_features = torch.cat([signals, delay_embed, sender_embedding, amp_embed, prob_embedding, symbol_embed], 2)
        input_features = input_features.reshape(batch_size*N_MIXER, 1+DELAY_EMB+SENDER_EMB+AMP_EMB+PROB_EMB+SYMBOL_EMB, N_FFT, n_frame)

        # input_features = torch.cat([signals, delay_embed, sender_embedding, amp_embed, prob_embedding], 2)
        # input_features = input_features.reshape(batch_size*N_MIXER, 1+DELAY_EMB+SENDER_EMB+AMP_EMB+PROB_EMB, N_FFT, n_frame)
        # input_features = input_features.reshape(batch_size * N_MIXER, 31, N_FFT, n_frame)
        encoding = self.encoder(input_features)
        
        encoding = encoding.reshape(batch_size*N_MIXER, N_FFT, n_frame)

        # mix the signals in time domain        
        time_signals = torch.istft(encoding, n_fft = N_FFT, return_complex=True).cuda()
        time_signals = time_signals.view(batch_size, N_MIXER, CHIRP_LEN)
                
        return time_signals

class UNet_without(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ComplexConv2d(1, 32, kernel_size=5,stride=1,padding = 0),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            ComplexConv2d(32, 32, kernel_size=5,stride=1,padding = 0),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            # ComplexMaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.layer3 = nn.Sequential(
            ComplexConv2d(32, 64, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexConv2d(64, 64, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            # ComplexAvgPool2d(kernel_size=2)
        )
        self.layer5 = nn.Sequential(
            ComplexConv2d(64, 128, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(128),
            ComplexReLU(),
            ComplexConv2d(128, 128, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(128),
            ComplexReLU()
        )
        self.layer7 = nn.Sequential(
            ComplexConv2d(128, 256, kernel_size=1,stride=1,padding = 0),
            ComplexBatchNorm2d(256),
            ComplexReLU(),
            ComplexConv2d(256, 256, kernel_size=1,stride=1,padding = 0),
            ComplexBatchNorm2d(256),
            ComplexReLU()
        )
        self.layer8 = nn.Sequential(
            ComplexConv2d(256, 128, kernel_size=1,stride=1,padding = 0),
            ComplexBatchNorm2d(128),
            ComplexReLU(),
            ComplexConv2d(128, 128, kernel_size=1,stride=1,padding = 0),
            ComplexBatchNorm2d(128),
            ComplexReLU()
        )
        self.layer6 = nn.Sequential(
            ComplexConv2d(256, 64, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexConv2d(64, 64, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(64),
            ComplexReLU()
            # ComplexConvTranspose2d(64,64,2)

        )
        self.layer4 = nn.Sequential(
            ComplexConv2d(128, 32, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            ComplexConv2d(32, 32, kernel_size=3,stride=1,padding = 0),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            # ComplexConvTranspose2d(32,32,2)
        )
        self.layer2 = nn.Sequential(
            ComplexConv2d(64, 1, kernel_size=5,stride=1,padding = 0),
            ComplexBatchNorm2d(1),
            ComplexReLU(),
            ComplexConv2d(1, 1, kernel_size=5,stride=1,padding = 0),
            ComplexBatchNorm2d(1),
            ComplexReLU()
            
        )

    def forward(self, x):
        # input: [batch_size*mix_num, 1+10+10, n_fft, n_frame]      [1signal + 10delay_embed + 10sender_embed]
        # output: [batch_size*mix_num, 1+10+10, n_fft, n_frame]

        # [N, 21,1024,129]->[N, 64,1024,129]
        
        layer1 = self.layer1(x)
        
        # layer1p = max_pool_2d(layer1,2)
        
        layer3 = self.layer3(layer1)
        
        layer5 = self.layer5(layer3)
        layer7 = self.layer7(layer5)
        layer8 = self.layer8(layer7)
        layer8_resize=upsample4(layer8, layer5.shape[2],layer5.shape[3])
        layer6 = self.layer6(torch.cat([layer8_resize, layer5], dim=1))
        layer6_resize=upsample4(layer6, layer3.shape[2],layer3.shape[3])
        layer4 = self.layer4(torch.cat([layer6_resize, layer3], dim=1))
        layer4_resize=upsample4(layer4, layer1.shape[2],layer1.shape[3])
        y = self.layer2(torch.cat([layer4_resize, layer1], dim=1))
        
        y_resize=upsample4(y, x.shape[2],x.shape[3])


        # layer1 = self.layer1(x)
        # [N,64,1024,129]->[N,1,1024,129]
        # x = self.layer4(layer1)
        return y_resize


class LoRaModel_without(torch.nn.Module):
    def __init__(self):
        super(LoRaModel_without, self).__init__()

        self.encoder = UNet_without()
    
    
    def forward(self, symbols, signals, delay_embed, sender_id, amps, probs):
        # symbols: [batch_size, N_MIXER, 1]
        # signal: [batch_size, N_MIXER, N_FFT, N_FRAME]
        # delay_embed: [batch_size, N_MIXER, DELAY_EMB]
        # sender_id: [batch_size, N_MIXER]
        # delays: [batch_size, N_MIXER]
        # amps: [batch_size, N_MIXER ,1]
        # probs: [1, N_SENDER]
        batch_size, _, _, n_frame = signals.shape
        signals = signals.unsqueeze(2)      # [batch_size, N_MIXER, 1, N_FFT, N_FRAME]

        # encode the input
        input_features = signals
        input_features = input_features.reshape(batch_size*N_MIXER, 1, N_FFT, n_frame)

        # input_features = torch.cat([signals, delay_embed, sender_embedding, amp_embed, prob_embedding], 2)
        # input_features = input_features.reshape(batch_size*N_MIXER, 1+DELAY_EMB+SENDER_EMB+AMP_EMB+PROB_EMB, N_FFT, n_frame)
        # input_features = input_features.reshape(batch_size * N_MIXER, 31, N_FFT, n_frame)
        encoding = self.encoder(input_features)
        
        encoding = encoding.reshape(batch_size*N_MIXER, N_FFT, n_frame)

        # mix the signals in time domain        
        time_signals = torch.istft(encoding, n_fft = N_FFT, return_complex=True).cuda()
        time_signals = time_signals.view(batch_size, N_MIXER, CHIRP_LEN)
                
        return time_signals
