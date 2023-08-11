import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import joblib
import glob
from typing import Iterable, Optional
import torch.nn as nn
import torch.nn.functional as Fun
import torch.multiprocessing
import torch.nn as nn
import shutil
from torch.utils.data import Dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
import timm
from timm.utils import accuracy
from util import misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from scipy.io import savemat
# from colorama import Fore, Back, Style


# class Encoder(nn.Module):
#     """编码器-解码器架构的基本编码器接口"""
#
#     def __init__(self, **kwargs):
#         super(Encoder, self).__init__(**kwargs)
#
#     def forward(self, X, *args):
#         raise NotImplementedError
#
#
# class Decoder(nn.Module):
#     """编码器-解码器架构的基本解码器接口"""
#
#     def __init__(self, **kwargs):
#         super(Decoder, self).__init__(**kwargs)
#
#     def init_state(self, enc_outputs, *args):
#         raise NotImplementedError
#
#     def forward(self, X, state):
#         raise NotImplementedError


# class EncoderDecoder(nn.Module):
#     """编码器-解码器架构的基类"""
#
#     def __init__(self, encoder, decoder, **kwargs):
#         super(EncoderDecoder, self).__init__(**kwargs)
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, enc_X, dec_X, *args):
#         enc_outputs = self.encoder(enc_X, *args)
#         dec_state = self.decoder.init_state(enc_outputs, *args)
#         return self.decoder(dec_X, dec_state)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_forward=True):
        super(BasicBlock, self).__init__()
        if is_forward:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            # 经过处理后的x要与x的维度相同(尺寸和深度)
            # 如果不相同，需要添加卷积+BN来变换为同一维度
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        else:
            output_padding = stride - 1
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3,
                                            stride=stride, padding=1, output_padding=output_padding, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            # 经过处理后的x要与x的维度相同(尺寸和深度)
            # 如果不相同，需要添加卷积+BN来变换为同一维度
            if in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_planes, self.expansion * planes, stride=stride,
                                       kernel_size=1, padding=0, output_padding=output_padding, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = Fun.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        aa = self.shortcut(x)
        out = out + aa
        out = Fun.relu(out)
        return out


# class Bottleneck(nn.Module):
#     # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion * planes,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )
#
#     def forward(self, x):
#         out = Fun.relu(self.bn1(self.conv1(x)))
#         out = Fun.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = Fun.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, inputsize, output_size, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = Fun.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = Fun.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


class MyResNet(nn.Module):
    def __init__(self, block, num_blocks, input_size, output_size, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 zero_init_last_bn=True, drop_rate=0.1):
        self.drop_rate = drop_rate
        super(MyResNet, self).__init__()
        self.in_planes = 64
        stem_chs = (3 * (64 // 4), 64)
        num_classes = output_size[1]

        self.conv1 = nn.Sequential(*[
            nn.Conv2d(input_size[1], stem_chs[0], 3, stride=2, padding=1, bias=False),
            norm_layer(stem_chs[0]),
            act_layer(inplace=True),
            nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
            norm_layer(stem_chs[1]),
            act_layer(inplace=True),
            nn.Conv2d(stem_chs[1], self.in_planes, 3, stride=1, padding=1, bias=False)])
        # self.conv1 = nn.Conv2d(input_size[1], 64, kernel_size=7,
        # stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.act1 = act_layer(inplace=True)
        # self.feature_info = [dict(num_chs=self.in_planes, reduction=2, module='act1')]
        # self.maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # channels = [64, 128, 256, 512]

        self.layer1 = self._make_layer(block, 64 * block.expansion, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 * block.expansion, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * block.expansion, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * block.expansion, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(BasicBlock, 256 * block.expansion, num_blocks[3], stride=2, is_forward=False)
        self.layer6 = self._make_layer(BasicBlock, 128 * block.expansion, num_blocks[2], stride=2, is_forward=False)
        self.layer7 = self._make_layer(BasicBlock, 64 * block.expansion, num_blocks[1], stride=2, is_forward=False)
        self.layer8 = self._make_layer(BasicBlock, 64 * block.expansion, num_blocks[0], stride=1, is_forward=False)

        self.conv2 = nn.Sequential(*[
            nn.ConvTranspose2d(self.in_planes, stem_chs[1], 3, stride=1, padding=1, output_padding=0, bias=False),
            norm_layer(stem_chs[1]),
            act_layer(inplace=True),
            nn.ConvTranspose2d(stem_chs[1], stem_chs[0], 3, stride=1, padding=1, output_padding=0, bias=False),
            norm_layer(stem_chs[0]),
            act_layer(inplace=True),
            nn.ConvTranspose2d(stem_chs[0], input_size[1], 3, stride=2, padding=1, output_padding=1, bias=False)])
        self.bn2 = norm_layer(input_size[1])
        self.act2 = act_layer(inplace=True)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.drop = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm([256, int(input_size[2] / 8), int(input_size[3] / 8)])
        self.ln2 = nn.LayerNorm([128, int(input_size[2] / 4), int(input_size[3] / 4)])
        self.ln3 = nn.LayerNorm([64, int(input_size[2] / 2), int(input_size[3] / 2)])
        self.ln4 = nn.LayerNorm([64, int(input_size[2] / 2), int(input_size[3] / 2)])
        # self.DNN = nn.Linear(512, 512)
        # self.layer5 = self._make_layer(block, 256, num_blocks[0], stride=2, is_forward=False)
        # self.layer6 = self._make_layer(block, 128, num_blocks[2], stride=2, is_forward=False)
        # self.layer7 = self._make_layer(block, 64, num_blocks[1], stride=2, is_forward=False)
        # self.layer8 = self._make_layer(block, 64, num_blocks[0], stride=1, is_forward=False)
        # self.avgPool1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgPool2 = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear_A1 = nn.Linear(512 * block.expansion, 512 * block.expansion, bias=True)
        # self.linear_A2 = nn.Linear(512 * block.expansion, output_size[1], bias=True)
        # self.linear_T1 = nn.Linear(512 * block.expansion, 512 * block.expansion, bias=True)
        # self.linear_T2 = nn.Linear(512 * block.expansion, output_size[1], bias=True)
        # self.out = torch.empty([1]+output_size)
        self.init_weights(zero_init_last_bn=zero_init_last_bn)

    def _make_layer(self, block, planes, num_blocks, stride, is_forward=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_forward))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def Ln(self, X):
    #     size = X.shape
    #     return nn.LayerNorm([size[1], size[2], size[3]])(X).to(device, non_blocking=True)

    def forward(self, x):
        x0 = self.act1(self.bn1(self.conv1(x)))
        # x = self.maxPool1(x)

        # Encoder
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder
        # dd = self.addNorm = AddNorm()
        x5 = self.layer5(x4)
        dd = self.ln1(self.drop(x5) + x3)
        x6 = self.layer6(dd)
        dd = self.ln2(self.drop(x6) + x2)
        x7 = self.layer7(dd)
        dd = self.ln3(self.drop(x7) + x1)
        x8 = self.layer8(dd)
        dd = self.ln4(self.drop(x8) + x0)
        out = self.act2(self.bn2(self.conv2(dd)))
        #
        #
        # xA = self.avgPool1(xA)
        # if self.drop_rate:
        #     xA = Fun.dropout(xA, p=float(self.drop_rate), training=self.training)
        # xA = torch.flatten(xA, 1)
        # xA = self.linear_A1(xA)
        # xA = self.linear_A2(xA).unsqueeze(-1)
        #
        # xT = self.layer5(x)
        # xT = self.layer6(xT)
        # xT = self.layer7(xT)
        # xT = self.layer8(xT)
        # xT = self.avgPool2(xT)
        # if self.drop_rate:
        #     xT = Fun.dropout(xT, p=float(self.drop_rate), training=self.training)
        # xT = torch.flatten(xT, 1)
        # xT = self.linear_T1(xT)
        # xT = self.linear_T2(xT).unsqueeze(-1)
        #
        # out = torch.stack([xA, xT], dim=1)
        # out = Fun.relu(self.bn1_T(self.conv1_T(out)))
        # out = self.max_pool(out)
        # out = torch.flatten(out, 1)
        # # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

    def init_weights(self, zero_init_last_bn=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()


class SampleDataset(Dataset):
    def __init__(self, file):
        file_data = np.load(file)
        self.samples = file_data['data']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# def ResNet50():
#     return MyResNet(Bottleneck, [3, 4, 6, 3])


def ResNet18(input_size, output_size, drop_rate):
    return MyResNet(BasicBlock, [2, 2, 2, 2], input_size, output_size, drop_rate=drop_rate)


class My_accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, device='cuda'):
        batch_size = target.size(0)
        # a = output[:, 0, :, :].squeeze().to(device, non_blocking=True)
        # b = target[:, 0, :, :].squeeze().to(device, non_blocking=True)
        # two_ab = torch.mul(torch.mul(a, b), 2)+
        # theta = torch.mul(torch.sub(output[:, 1, :, :].squeeze(), target[:, 1, :, :].squeeze()), torch.pi).to(device, non_blocking=True)
        # c_2 = torch.pow(a, 2) + torch.pow(b, 2) - torch.mul(two_ab, torch.cos(theta))
        # # c = torch.sqrt(c_2+Epsilon)
        # # rmse = torch.mean(c)
        # rmse = torch.sum(c_2)
        # Epsilon = torch.mul(torch.ones(a.shape), 1E-10).to(device, non_blocking=True)
        a = output.squeeze().to(device, non_blocking=True)
        b = target.squeeze().to(device, non_blocking=True)
        a_b = torch.sub(a, b)
        a_b_2 = torch.pow(a_b, 2)
        c = torch.mean(a_b_2)
        d = torch.mul(c, 2)  # 双通道求均值，需要补偿2
        # Epsilon = torch.mul(torch.ones(d.shape), 1E-10).to(device, non_blocking=True)
        rmse = torch.pow(d, 0.5)

        # a0 = a[0, :, :]
        # a1 = a[1, :, :]
        # b0 = b[0, :, :]
        # b1 = b[1, :, :]
        a0 = torch.sub(a[0, :, :], torch.mean(a[0, :, :]))
        a1 = torch.sub(a[1, :, :], torch.mean(a[1, :, :]))
        b0 = torch.sub(b[0, :, :], torch.mean(b[0, :, :]))
        b1 = torch.sub(b[1, :, :], torch.mean(b[1, :, :]))
        real = torch.sum(torch.add(torch.mul(a0, b0), torch.mul(a1, b1)))
        img = torch.sum(torch.sub(torch.mul(a1, b0), torch.mul(a0, b1)))
        Am_2 = torch.add(torch.pow(real, 2), torch.pow(img, 2))
        A = torch.pow(Am_2, 0.5)

        Aa = torch.pow(torch.sum(torch.add(torch.pow(a0, 2), torch.pow(a1, 2))), 0.5)
        Ab = torch.pow(torch.sum(torch.add(torch.pow(b0, 2), torch.pow(b1, 2))), 0.5)
        ro = torch.div(A, torch.mul(Aa, Ab))
        return rmse, ro


@torch.no_grad()
def evaluate(dataset_eva, model, epoch, device):
    criterion = My_accuracy()
    # switch to evaluation mode
    model.eval()
    loss_list = []
    ro_list = []
    for idx, eva_data in enumerate(dataset_eva):
        eva_data = np.array(eva_data)

        input = torch.from_numpy(eva_data[0:2, :, :]).unsqueeze(0).type(torch.FloatTensor)
        label = torch.from_numpy(eva_data[3:5, :, :]).unsqueeze(0).type(torch.FloatTensor)
        input = input.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(input)
        # dd=output.squeeze().numpy()[1]
        criterion = My_accuracy()
        loss, ro = criterion(output, label)
        print("No.{} Evaluate: Loss is {} and ro is {} ".format(idx, loss, ro))
        loss_list.append(loss)
        ro_list.append(ro)
    # loss_list.cpu().detach().numpy()
    # ro_list.cpu().detach().numpy()
    # lossSaved = np.array(loss_list)
    # roSaved = np.array(ro_list)
    path = args.main_file + '/Evaluate/'
    loss_list = torch.tensor(loss_list, device='cpu')
    ro_list = torch.tensor(ro_list, device='cpu')
    np.savetxt(path + '{}-Evaluate_Loss.csv'.format(epoch), loss_list, delimiter=',')
    np.savetxt(path + '{}-Evaluate_ro.csv'.format(epoch), ro_list, delimiter=',')



def build_dataset(args, mode):
    # path = os.path.join(args.root_path, 'train' if is_train else 'test')
    if mode == 'train':
        dataset = SampleDataset(args.train_path)
    elif mode == 'test':
        dataset = SampleDataset(args.test_path)
    elif mode == 'evaluate':
        dataset = SampleDataset(args.eva_path)
    else:
        assert "mode is worong"
    return dataset


def draw(args, output, input, label):
    out = output.cpu().detach().numpy()
    # A=out[0,0,:,:]
    # T=np.cos(out[0,1,:,:])
    # AT=A*T
    AT = out[0, 0, :, 0]
    t_line = np.arange(-args.TpLen / 2, args.TpLen / 2) / args.Fs

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(t_line, AT)
    plt.xlabel('时间')
    plt.ylabel('振幅')
    plt.title('out')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    out = input.cpu().detach().numpy()
    # A=out[0,0,:,:]
    # T=np.cos(out[0,1,:,:])
    # AT=A*T
    AT = out[0, 0, :, 0]
    plt.subplot(3, 1, 2)
    plt.plot(t_line, AT)
    plt.xlabel('时间')
    plt.ylabel('振幅')
    plt.title('input')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    out = label.cpu().detach().numpy()
    # A=out[0,0,:,:]
    # T=np.cos(out[0,1,:,:])
    # AT=A*T
    AT = out[0, 0, :, 0]
    plt.subplot(3, 1, 3)
    plt.plot(t_line, AT)
    plt.xlabel('时间')
    plt.ylabel('振幅')
    plt.title('label')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None,
                    lossSaved=None,
                    roSaved=None):
    model.train(True)

    print_freq = 2

    accum_iter = args.accum_iter  # 每隔accum_iter做一次梯度更新

    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples) in enumerate(data_loader):
        input = samples[:, 0:2, :, :].type(torch.FloatTensor)
        label = samples[:, 3:5, :, :].type(torch.FloatTensor)
        input = input.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        outputs = model(input)

        # warmup_lr = args.lr*(min(1.0, epoch/2.))
        warmup_lr = args.lr * math.pow(0.99999, epoch)
        optimizer.param_groups[0]["lr"] = warmup_lr

        loss, ro = criterion(outputs, label)
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)  # 反向传播实现loss.backward和优化器更新optimizer.step()
        loss_value = loss.item()
        ro_value = ro.item()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if not math.isfinite(loss_value):
            print("Loss is {}, ro is {}, stopping training".format(loss_value, ro_value))
            draw(args=args, output=outputs, input=input, label=label)
            sys.exit(1)

        if log_writer is not None:
            # step = int(epoch * len(data_loader) + data_iter_step)
            # log_writer.add_scalar('loss', loss_value, step)
            # log_writer.add_scalar('lr', warmup_lr, step)
            # lossSaved.append([step, loss_value])
            print(f"Epoch: {epoch}, Step: {data_iter_step}, Loss: {loss}, ro: {ro}, Lr:{warmup_lr}")
    return lossSaved, roSaved
    # if data_iter_step == 225:
    #     return outputs


def real2complex(real_data, subvalue, divvalue):
    mulvalue = divvalue * (real_data[0, 0, :, :] + 1j * real_data[0, 1, :, :])
    complex_data = np.squeeze(mulvalue + (subvalue + 1j * subvalue))
    return complex_data


def to_dB(value, type='power'):  # 真值 --> dB
    if type == 'power':
        dB = 10 * np.log10(value+1e-20)
    else:
        dB = 20 * np.log10(value+1e-20)
    return dB


def CFAR_2D(sig):
    # 平方率检波后处理
    # sig = np.absolute(sig)
    Pfa = 1E-9
    refCellNum = 16
    protectCellNum = 8
    DSNR = 13.2
    DSNR = 10 ** (DSNR / 20)
    K = refCellNum * (Pfa ** (-1 / refCellNum) - 1)
    [n1, n2] = sig.shape
    CFAR_flag_output = np.zeros((n1, n2))
    CFAR_out = np.zeros((n1, n2))
    CFAR_VN = np.zeros((n1, n2))
    shang = refCellNum
    shang_pro = protectCellNum
    xia = refCellNum
    xia_pro = protectCellNum
    zuo = refCellNum
    zuo_pro = protectCellNum
    you = refCellNum
    you_pro = protectCellNum
    sig_2 = np.append(np.append(sig, sig, axis=1), sig, axis=1)
    for m in np.linspace(start=shang + shang_pro, stop=n1 - xia - xia_pro - 1,
                         num=(n1 - xia - xia_pro) - (shang + shang_pro)):
        m = int(m)
        for n in np.linspace(start=0, stop=n2 - 1, num=n2):
            n = int(n)
            data_r_shang = sum(sig[m - shang - shang_pro:m - shang_pro - 1, n])
            v1 = data_r_shang / shang
            data_r_xia = sum(sig[m + xia_pro + 1:m + xia + xia_pro, n])
            v2 = data_r_xia  / xia
            if (0 <= n) and (n <= zuo + zuo_pro):
                data_f_zuo = sum(sig_2[m, n2 + n - zuo - zuo_pro:n2 + n - zuo_pro - 1])
                v3 = data_f_zuo / zuo
                data_f_you = sum(sig[m, n + you_pro + 1:n + you + you_pro])
                v4 = data_f_you / you
            elif (zuo + zuo_pro + 1 <= n) and (n <= n2 - you - you_pro):
                data_f_zuo = sum(sig[m, n - zuo - zuo_pro:n - zuo_pro - 1])
                v3 = data_f_zuo / zuo
                data_f_you = sum(sig[m, n + you_pro + 1:n + you + you_pro]);
                v4 = data_f_you  / you
            elif (n2 - you - you_pro + 1 <= n) and (n <= n2 - 1):
                data_f_zuo = sum(sig[m, n - zuo - zuo_pro:n - zuo_pro - 1])
                v3 = data_f_zuo / zuo
                data_f_you = sum(sig_2[m, n + you_pro + 1:n + you + you_pro])
                v4 = data_f_you  / you
            CFAR_VN[m, n] = (np.mean([v1, v2, v3, v4])) * K
            if sig[m, n] >= CFAR_VN[m, n]:
                CFAR_flag_output[m, n] = 1
                CFAR_out[m, n] = sig[m, n]
    return CFAR_flag_output, CFAR_VN


def Pulse_compression_MTI(data, LFM):
    Nfft_f = LFM.shape[1] * 1
    Nfft_r = LFM.shape[0] + data.shape[0] - 1
    f_LFM = np.fft.fft(LFM, n=Nfft_r, axis=0)
    f_data = np.fft.fft(data, n=Nfft_r, axis=0)
    # 脉压
    fPC_data = np.multiply(f_data, f_LFM.conjugate())
    PC_result_data = np.fft.ifft(fPC_data, n=Nfft_r, axis=0)
    # 积累
    PC_result_data = np.fft.fft(PC_result_data, n=Nfft_f, axis=1)
    PC_result_data = np.fft.ifftshift(PC_result_data)
    return PC_result_data


def PC_CFAR(result, constant):
    # t_line = result[0]
    LFM = real2complex(result[1], constant[2], constant[3])
    Input = real2complex(result[2], constant[0], constant[1])
    Output = real2complex(result[3], constant[0], constant[1])
    # savemat("Temporary_data/try.mat", {'LFM': LFM, 'Input': Input, 'Output': Output})
    # 脉压积累
    PC_result_input = Pulse_compression_MTI(Input, LFM)
    PC_result_output = Pulse_compression_MTI(Output, LFM)
    # 取模
    PC_result_input = pow(np.absolute(PC_result_input), 2)
    PC_result_output = pow(np.absolute(PC_result_output), 2)
    # CFAR
    CFAR_flag_output, CFAR_VN = CFAR_2D(PC_result_output)
    if sum(sum(CFAR_flag_output)) != 0:
        CFAR_flag = 1
    else:
        CFAR_flag = 0
    # to dB
    PC_result_input = to_dB(PC_result_input, type='power')
    PC_result_output = to_dB(PC_result_output, type='power')
    CFAR_VN= to_dB(CFAR_VN, type='power')

    # Nfft_r = LFM.shape[0] + len(t_line) - 1
    # Nfft_f = LFM.shape[1] * 2
    # f_LFM = np.fft.fft(LFM, n=Nfft_r, axis=0)
    # f_Input = np.fft.fft(Input, n=Nfft_r, axis=0)
    # f_Output = np.fft.fft(Output, n=Nfft_r, axis=0)
    # fPC_input = np.multiply(f_Input, f_LFM.conjugate())
    # fPC_output = np.multiply(f_Output, f_LFM.conjugate())
    # # 脉压
    # PC_result_input = np.fft.ifft(fPC_input, n=Nfft_r, axis=0)
    # PC_result_output = np.fft.ifft(fPC_output, n=Nfft_r, axis=0)
    # # 积累
    # PC_result_input = np.fft.fft(PC_result_input, n=Nfft_f, axis=1)
    # PC_result_output = np.fft.fft(PC_result_output, n=Nfft_f, axis=1)
    # # PC_result_input = np.flip(PC_result_input, axis=1)
    # # PC_result_output = np.flip(PC_result_output, axis=1)
    # PC_result_input = np.fft.ifftshift(PC_result_input)
    # PC_result_output = np.fft.ifftshift(PC_result_output)
    # savemat("Temporary_data/result.mat", {'resultInput': PC_result_input, 'resultOutput': PC_result_output, 'CFAR_flag_output': CFAR_flag_output, 'CFAR_VN': CFAR_VN})
    return PC_result_input, PC_result_output, CFAR_flag, CFAR_flag_output, CFAR_VN


def main(args, mode='train', test_data='', test_idx=None, model=None, epoch_idx=None, device=device):
    if mode == 'train':
        # 构建数据批次
        dataset_train = build_dataset(args=args, mode='train')
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        # args.eva_path = args.main_file + "\Evaluate" + "\data_test.npz"
        dataset_eva = build_dataset(args=args, mode='evaluate')
        # sampler_eva = torch.utils.data.RandomSampler(dataset_eva)
        # # sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        # data_loader_eva = torch.utils.data.DataLoader(
        #     dataset_train, sampler=sampler_eva,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     pin_memory=args.pin_mem,
        #     drop_last=True,
        # )


        # if is_train=False:
        #     dataset_test = build_dataset(args=args, is_train=False)
        #     sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        #     data_loader_test = torch.utils.data.DataLoader(
        #         dataset_test, sampler=sampler_test,
        #         batch_size=1,
        #         # batch_size=1
        #         num_workers=args.num_workers,
        #         pin_memory=args.pin_mem,
        #         drop_last=False,
        #     )
        data_size = dataset_train.samples.shape
        x = torch.randn(1, 2, data_size[2], data_size[3])
        model = ResNet18(input_size=[1, 2, data_size[2], data_size[3]], output_size=[2, data_size[2], data_size[3]], drop_rate=args.drop_out)
        # y = model(x)
        # print(y.size())
        # model = timm.create_model('resnet18', pretrained=True, num_classes=args.TpLen, drop_rate=args.drop_out, drop_path_rate=0.1)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of trainable params (M): %.2f' % (n_parameters / 1.e6))

        criterion = My_accuracy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        loss_scaler = NativeScaler()

        # misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, loca='cuda')
        # args.resume = args.main_file +'/epochs/model/checkpoint-399.pth'
        # checkpoint = torch.load(args.resume)
        # model.load_state_dict(checkpoint['model'])

        lossSaved = []
        roSaved = []
        for epoch in range(0, args.epochs):
            print(f"Epoch {epoch}")
            print(f"length of data_loader_train is {len(data_loader_train)}")
            model = model.cuda()
            model.train()
            if epoch % 1 == 0:
                print("Evaluating...")
                model.eval()
                with torch.no_grad():
                    evaluate(dataset_eva, model, epoch, device)
                model.train()

            # 然后进行训练
            lossSaved = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch,
                loss_scaler, None,
                log_writer=log_writer,
                args=args,
                lossSaved=lossSaved,
                roSaved=roSaved
            )

            if (epoch+1) % 50 == 0:
                print("Saving checkpoints...")
                misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        # lossSaved = np.array(lossSaved)
        # loss_path = "Power_"+args.type + '/loss/'
        # os.makedirs(loss_path, exist_ok=True)
        # np.savetxt(loss_path+'Loss.csv', lossSaved, delimiter=',')
        # joblib.dump(lossSaved, 'output_dir_pretrained/loss.pkl')

    else:
        input = torch.from_numpy(test_data[0:2, :, :]).unsqueeze(0).type(torch.FloatTensor)
        label = torch.from_numpy(test_data[3:5, :, :]).unsqueeze(0).type(torch.FloatTensor)
        input = input.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(input)
        # dd=output.squeeze().numpy()[1]
        criterion = My_accuracy()
        loss_old, ro_old = criterion(input, label)
        loss, ro = criterion(output, label)
        # print(f"Test data No.{test_idx}, Loss is {loss}")
        t_line = np.arange(-args.TpLen / 2, args.TpLen / 2) / args.Fs
        result = [t_line]
        for data in [label, input, output]:
            data = data.cpu().detach().numpy()
            result.append(data)
        # draw(args=args, output=output, input=input, label=label)
        return result, \
               loss.cpu().detach().numpy(), \
               ro.cpu().detach().numpy(), \
               loss_old.cpu().detach().numpy(), \
               ro_old.cpu().detach().numpy(),

        # print(f"image path is {test_image_path}")
        # print(f"score is {score.item()}, class id is {class_idx.item()}, class name is {list(class_dict.keys())[list(class_dict.values()).index(class_idx)]}")


def clean_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

def single_calculate(args):
    mode = 'test'
    dataset_test = build_dataset(args=args, mode='test').samples.tolist()
    data_size = np.array(dataset_test[0]).shape
    loss_scaler = NativeScaler()
    model = ResNet18(input_size=[1, 2, data_size[1], data_size[2]],
                     output_size=[2, data_size[1], data_size[2]], drop_rate=args.drop_out)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    i = args.epochs - 1
    args.resume = 'Temporary_compute/model/checkpoint-{}.pth'.format(i)
    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
    model.eval()
    for idx, test_data in enumerate(
            tqdm(dataset_test, desc="Test_data的进度：", colour='blue')):
        test_data = np.array(test_data)
        constant = [test_data[2, 0, 0], test_data[2, 1, 0],
                    test_data[5, 0, 0], test_data[5, 1, 0]]
        result, loss, ro, loss_old, ro_old = main(args, mode=mode, test_data=test_data, test_idx=idx,
                                                  model=model,
                                                  epoch_idx=i)
        PC_result_input_dB, PC_result_output_dB, flag, CFAR_flag_output, CFAR_VN = PC_CFAR(result, constant)
        # savemat("Temporary_data/try.mat", {'PC_result_input_dB': PC_result_input_dB, 'PC_result_output_dB': PC_result_output_dB, 'CFAR_flag_output': CFAR_flag_output, 'CFAR_VN': CFAR_VN})
        if flag == 1 and idx >= 500:
            savemat("Temporary_data/try.mat",
                    {'PC_result_input_dB': PC_result_input_dB, 'PC_result_output_dB': PC_result_output_dB,
                     'CFAR_flag_output': CFAR_flag_output, 'CFAR_VN': CFAR_VN})
            dd=1



if __name__ == '__main__':
    args = joblib.load('parameter_setting.pkl')
    args.num_workers = 3
    # args = joblib.load('Temporary_compute/setting/parameter_setting.pkl')
    # args.test_path = 'Temporary_compute/data_test.npz'
    # single_calculate(args)
    args.train_path = args.main_file + '/Data_set/total_data_train.npz'
    # mode = 'train'  #  train or test
    for mode in ['train', 'test']:
        if mode == 'train':
            print(f"{mode} mode...")
            main(args, mode=mode)
            print("Training have finished.")
        else:
            for index_SNR in tqdm(range(int(args.SNR.shape[0])), desc="SNR的进度：", colour='green'):
                # for index_SNR in tqdm(np.array([11, 0]), desc="SNR的进度：", colour='green'):
                SNR_path = args.main_file + "/Data_set/SNR_%.0f" % int(args.SNR[index_SNR])
                # SNR_path = args.main_file + "/Data_set/SNR_%.0f" % int(args.SNR[-1])
                loss_SNR = []
                ro_SNR = []
                loss_old_SNR = []
                ro_old_SNR = []
                Pd_SNR = []
                for index_JSR in tqdm(range(int(args.JSR.shape[0])), desc="JSR的进度：", colour='red'):
                    JSR_path = "/JSR_%.0f" % int(args.JSR[index_JSR])
                    # JSR_path = "/JSR_%.0f" % int(args.JSR[0])
                    args.test_path = SNR_path + JSR_path + "/data_test.npz"
                    dataset_test = build_dataset(args=args, mode='test').samples.tolist()
                    data_size = np.array(dataset_test[0]).shape
                    model = ResNet18(input_size=[1, 2, data_size[1], data_size[2]],
                                     output_size=[2, data_size[1], data_size[2]], drop_rate=args.drop_out)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    loss_scaler = NativeScaler()
                    i = (args.epochs - 1)
                    args.resume = args.main_file + ('/epochs/model/checkpoint-{}.pth'.format(399))
                    # result_path = "Power_" + args.type + ('/epochs/results/epoch-{}/'.format(i)) + \
                    #               ('SNR_%.0f/'.format(int(args.SNR[index_SNR]))) + \
                    #               # ('JSR_%.0f'.format(int(args.JSR[index_JSR])))
                    # PC_input_path = SNR_path + JSR_path + "/PC_input"
                    # PC_output_path = SNR_path + JSR_path + "/PC_output"
                    # # os.makedirs(result_path, exist_ok=True)
                    # clean_path(PC_input_path)
                    # clean_path(PC_output_path)
                    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
                    model.eval()
                    model = model.cuda()
                    loss_JSR = []
                    ro_JSR = []
                    loss_old_JSR = []
                    ro_old_JSR = []
                    Detection_times = 0
                    for idx, test_data in enumerate(
                            tqdm(dataset_test, desc="Test_data的进度：", colour='blue')):
                        test_data = np.array(test_data)
                        constant = [test_data[2, 0, 0], test_data[2, 1, 0],
                                    test_data[5, 0, 0], test_data[5, 1, 0]]
                        result, loss, ro, loss_old, ro_old = main(args, mode=mode, test_data=test_data, test_idx=idx,
                                                                  model=model,
                                                                  epoch_idx=i)
                        PC_result_input_dB, PC_result_output_dB, flag, CFAR_flag_output, CFAR_VN = PC_CFAR(result, constant)
                        # savemat("Temporary_data/result.mat", {'PC_result_input_dB': PC_result_input_dB, 'PC_result_output_dB': PC_result_output_dB, 'CFAR_flag_output': CFAR_flag_output, 'CFAR_VN': CFAR_VN})
                        if flag == 1:
                            Detection_times = Detection_times + 1
                        # result_name = ("/No.%.0f - " % idx) + ("loss_is_%.5f - " % loss) + ("ro_is_%.5f - " % ro) + (
                        #         "flag_is_%.0f.csv" % flag)
                        # np.savetxt(PC_input_path + result_name, PC_result_input_dB, delimiter=',')
                        # np.savetxt(PC_output_path + result_name, PC_result_output_dB, delimiter=',')
                        loss_JSR.append(loss)
                        ro_JSR.append(ro)
                        loss_old_JSR.append(loss_old)
                        ro_old_JSR.append(ro_old)
                    # np.savetxt(result_path + result_name, result, delimiter=',')
                    # np.savetxt(SNR_path + JSR_path + '/detection_probability.csv',
                    #            Detection_times / args.test_data_number, delimiter=',')
                    loss_SNR.append(np.mean(loss_JSR))
                    ro_SNR.append(np.mean(ro_JSR))
                    loss_old_SNR.append(np.mean(loss_old_JSR))
                    ro_old_SNR.append(np.mean(ro_old_JSR))
                    Pd_SNR.append(Detection_times / args.test_data_number)
                    tqdm.write("\n SNR:%.0f - " % int(args.SNR[index_SNR]) +
                               "JSR:%.0f - " % int(args.JSR[index_JSR]) +
                               "Loss:%.4f - " % np.mean(loss_JSR) +
                               "ro:%.4f - " % np.mean(ro_JSR) +
                               "Loss_Old:%.4f - " % np.mean(loss_old_JSR) +
                               "ro_Old:%.4f " % np.mean(ro_old_JSR) +
                               "Pd:%.4f " % (Detection_times / args.test_data_number)
                               )
                np.savetxt(SNR_path + '/Pd_SNR.csv', Pd_SNR, delimiter=',')
                np.savetxt(SNR_path + '/loss_in_SNR.csv', loss_SNR, delimiter=',')
                np.savetxt(SNR_path + '/ro_in_SNR.csv', ro_SNR, delimiter=',')
            # np.savetxt(SNR_path + '/loss_old_in_SNR.csv', loss_old_SNR, delimiter=',')
            # np.savetxt(SNR_path + '/ro_old_in_SNR.csv', ro_old_SNR, delimiter=',')
            print("Testing have finished.")

            # out=train_stats.cpu().detach().numpy()
            # A=out[0,0,:,:]
            # T=np.cos(out[0,1,:,:])
            # AT=A*T
            # t_line = np.arange(-args.TpLen / 2, args.TpLen / 2) / args.Fs
            # plt.plot(t_line, AT)
            # plt.xlabel('时间')
            # plt.ylabel('振幅')
            # plt.title('线性chirp仿真')
            # plt.rcParams['font.sans-serif']=['SimHei']
            # plt.rcParams['axes.unicode_minus'] = False
            # plt.show()
            # dd=1
