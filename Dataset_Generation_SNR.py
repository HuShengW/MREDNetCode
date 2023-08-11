import sys

import matplotlib.pyplot as plt
import os
import numpy as np
import math
import random
import argparse
from tqdm import tqdm
import joblib
import time
import torch

import scipy.io as scio


def parameter_setting():
    parser = argparse.ArgumentParser('Basic parameters', add_help=False)
    parser.add_argument('--carrier_frequency', default=4E9, type=int, help='载频')
    parser.add_argument('--C', default=299792458, type=float, help='光速')
    parser.add_argument('--B', default=5E+6, type=float, help='带宽')
    parser.add_argument('--prf', default=2E+3, type=float, help='重复频率')
    parser.add_argument('--Tp', default=160E-6, type=float, help='脉宽长度')
    parser.add_argument('--num_pluses', default=64, type=int, help='脉冲个数')
    parser.add_argument('--velocity', default=10, type=float, help='目标速度')
    parser.add_argument('--v_sigma', default=0, type=float, help='量测速度误差')
    args = parser.parse_args()
    Fs = 1 * args.B
    parser.add_argument('--Fs', default=Fs, type=float, help='采样频率')
    TpLen = int(args.Tp * Fs)
    parser.add_argument('--TpLen', default=TpLen, type=int, help='脉宽点数')
    miu = args.B / args.Tp
    parser.add_argument('--miu', default=miu, type=float, help='LFM线性调制斜率')
    parser.add_argument('--miu_max', default=3E+10, type=float, help='LFM线性调制斜率调节范围')
    parser.add_argument('--miu_min', default=3E+9, type=float, help='LFM线性调制斜率调节范围')
    r_resolution = args.C / 2 / args.B
    parser.add_argument('--r_resolution', default=r_resolution, type=float, help='分辨力')
    parser.add_argument('--type', default='SMSP-SMSP', help='干扰类型')

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--input_size', default=TpLen, type=int,
                        help='images input size')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--root_path', default='./', help='根目录，相对路径')
    parser.add_argument('--output_dir', default='./output_dir_pretrained',
                        help='模型存储路径')
    parser.add_argument('--log_dir', default='./output_dir_pretrained',
                        help='日志路径')

    parser.add_argument('--num_workers', default=5, type=int, help='工作核心数')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    train_data_number = 2E4
    parser.add_argument('--train_data_number', default=train_data_number, type=int, help='训练数据量')
    parser.add_argument('--test_data_number', default=1E3, type=int, help='测试数据量')
    parser.add_argument('--epochs', default=400, type=int, help='epoch数目')
    parser.add_argument('--batch_size', type=int, default=5, help='batch的尺寸')
    parser.add_argument('--SNR', default=np.linspace(-60, -15, 10), help='信噪比')
    parser.add_argument('--JSR', default=np.linspace(-10, 45, 12), help='干信比')
    # parser.add_argument('--SNR', default=np.linspace(-45, 0, 10), help='信噪比')
    # parser.add_argument('--JSR', default=np.linspace(-10, 45, 12), help='干信比')
    # parser.add_argument('--SNR', default=np.array([0]), help='信噪比')
    # parser.add_argument('--JSR', default=np.array([-1000]), help='信噪比')

    # parser.add_argument('--train_data_number', default=100, type=int, help='训练数据量')
    # parser.add_argument('--test_data_number', default=10, type=int, help='测试数据量')
    # parser.add_argument('--epochs', default=50, type=int, help='epoch数目')
    # parser.add_argument('--SNR', default=np.linspace(-60, 0, 12), help='信噪比')
    # parser.add_argument('--JSR', default=np.linspace(-10, 80, 19), help='干信比')

    parser.add_argument('--lr', type=float, default=0.2 / train_data_number, metavar='LR')
    parser.add_argument('--drop_out', type=float, default=0.1, metavar='drop_out')
    return parser


def generate_signal(LFM, args):  #
    Doppler_f = 2 * args.velocity * args.carrier_frequency / args.C
    N_Tr = np.dot(1 / args.prf, np.arange(0, args.num_pluses, 1))
    N_Theta = np.expand_dims(np.exp(1j * 2 * np.pi * Doppler_f * N_Tr), 0)
    LFM = np.expand_dims(LFM, 1)
    tran_sig = np.dot(LFM, N_Theta)
    return tran_sig


def add_noise_to_signal(signal, SNR):  # 为signal添加SNR db的噪声
    size = signal.shape
    noise = (np.random.randn(size[0], size[1]) + 1j * np.random.randn(size[0], size[1])) / (math.pow(2, 0.5))
    noise_power = to_value(-SNR, type='power')
    echo = math.sqrt(noise_power) * noise + signal
    return echo


def add_jamming_to_signal(signal, JSR, args, type='random'):
    size = signal.shape
    if type == 'random':
        jamming = signal
        jamming_min_len = int(0.05 * size[0])  # 最短干扰长度（/个采样点）
        start_position = random.randint(0, size[0] * 0.95)
        end_position = random.randint(start_position + jamming_min_len, size[0])
        jamming[start_position:end_position, :] = add_noise_to_signal(signal[start_position:end_position, :], JSR)
        position = [start_position, end_position]
    elif type == 'ISRJ':
        # cost_time = random.randint(10, 40)  # 个Fs的干扰机收发时间
        # jamming_times = random.randint(2, 8)  # 干扰采样周期数
        cost_time = 40  # 个Fs的干扰机收发时间
        jamming_times = 4  # 干扰采样周期数
        step = (jamming_times + 1) * cost_time
        jamming_tran = np.zeros((size[0] + step - size[0] % step, size[1]), dtype=complex)
        start_position = np.arange(0, size[0] - 1, step)
        end_position = start_position + cost_time * np.ones(start_position.shape)
        for i in range(start_position.size):
            if end_position[i] <= size[0]:
                sample = signal[int(start_position[i]):int(end_position[i]), :]
                jamming_tran[int(end_position[i]):(int(end_position[i]) + jamming_times * cost_time), :] = np.kron(
                    np.ones((jamming_times, 1)), sample)
            else:
                break
        add_jamming = jamming_tran[0:size[0], :]
        jamming_power = to_value(JSR, type='power')
        jamming = signal + math.sqrt(jamming_power) * add_jamming
    elif type == 'INUSRJ':
        Kmin = 0.25  # 0<Kmin<1 保留Kmin * size[0]的数据作为采样最小数据量
        jamming_cycles = random.randint(2, 8)  # 干扰采样周期数
        min_len = math.floor(size[0] * Kmin / jamming_cycles)
        rand_nums = np.random.rand(jamming_cycles)
        rand_p = np.floor(
            np.multiply(rand_nums / sum(rand_nums), size[0] * (1 - Kmin)) + size[0] * Kmin / jamming_cycles)
        rand_p[-1] = int(rand_p[-1] + (size[0] - sum(rand_p)))
        start = 0
        add_jamming = np.zeros((size[0], size[1]), dtype=complex)
        for keep_time in rand_p:
            try:
                sample_time = random.randint(int(min_len / 2), int(keep_time / 2))
            except:
                sample_time = random.randint(int(min_len / 2), int(keep_time / 2))
            retran_num = math.ceil(keep_time / sample_time) - 1
            sample = signal[start:(start + sample_time)]
            jamming_tran = np.kron(np.ones((retran_num, 1)), sample)
            add_jamming[(start + sample_time):(start + int(keep_time)), :] = jamming_tran[
                                                                             0:int(keep_time - sample_time), :]
            # position = [start+sample_time, start+int(keep_time)]
            # print([start, start+int(keep_time)])
            start = start + int(keep_time)
        jamming_power = to_value(JSR, type='power')
        jamming = signal + math.sqrt(jamming_power) * add_jamming
        # for i in range(start_position.size):
        #     sample = signal[int(start_position[i]):int(end_position[i]), :]
        #     jamming_tran[int(end_position[i]):(int(end_position[i])+jamming_cycles * cost_time), :] = np.kron(np.ones((jamming_cycles, 1)), sample)
        # add_jamming = jamming_tran[0:size[0], :]
        # jamming_power = to_value(JSR, type='power')
        # jamming = signal + jamming_power*add_jamming
    elif type == 'SMSP':
        # jamming_times = random.randint(2, 20)
        jamming_times = 5
        cost_time = 0  # 个Fs的干扰机收发时间
        Tp_j = int(size[0] / jamming_times)
        miu_j = args.miu * jamming_times
        t_line_j = np.arange(-Tp_j / 2, Tp_j / 2) / args.Fs
        sample = np.exp(1j * np.pi * miu_j * t_line_j * t_line_j)
        sample_matrix = generate_signal(sample, args=args)
        samples = np.kron(np.ones((jamming_times, 1)), sample_matrix)
        add_jamming = np.zeros((size[0], size[1]), dtype=complex)
        add_jamming[cost_time:samples.shape[0], :] = samples[0:(size[0] - cost_time), :]
        jamming_power = to_value(JSR, type='power')
        jamming = signal + math.sqrt(jamming_power) * add_jamming
    else:
        pass
    return jamming, add_jamming


def to_dB(value, type='power'):  # 真值 --> dB
    if type == 'power':
        dB = 10 * np.log10(value)
    else:
        dB = 20 * np.log10(value)
    return dB


def to_value(dB, type='power'):  # dB --> 真值
    if type == 'power':
        value = math.pow(10, dB / 10)
    else:
        value = math.pow(10, dB / 20)
    return value


def data_change(data, Theoretical=0):  # dB --> 真值
    size = data.shape
    data_changed = np.zeros((3, size[0], size[1]))
    # data_tensor = torch.zeros(2, len(data), 1)
    # data_changed[0, :, :] = np.absolute(data)/(np.absolute(data).max())
    # data_changed[1, :, :] = np.angle(data)/math.pi
    data_r = data.real
    data_i = data.imag
    max_value = max(data_r.max(), data_i.max())
    min_value = min(data_r.min(), data_i.min())
    data_changed[0, :, :] = (data_r - min_value) / (max_value - min_value)
    data_changed[1, :, :] = (data_i - min_value) / (max_value - min_value)
    data_changed[2, 0, 0] = min_value
    data_changed[2, 1, 0] = (max_value - min_value)
    return data_changed


def PC_data(args):
    miu = 1.5E+10
    t_line = np.arange(-args.TpLen / 2, args.TpLen / 2) / args.Fs
    LFM = np.exp(1j * np.pi * miu * t_line * t_line)
    tran_sig = generate_signal(LFM, args=args)
    echo = add_noise_to_signal(tran_sig, args.SNR)
    SMSP, _ = add_jamming_to_signal(echo, JSR=args.JSR, type='SMSP', args=args)  # 添加干扰信号，满足JSR
    ISRJ, _ = add_jamming_to_signal(echo, JSR=args.JSR, type='ISRJ', args=args)  # 添加干扰信号，满足JSR
    INUSRJ, _ = add_jamming_to_signal(echo, JSR=args.JSR, type='INUSRJ', args=args)  # 添加干扰信号，满足JSR
    Nfft = len(LFM) + len(t_line) - 1
    f_LFM = np.fft.fft(LFM, n=Nfft)
    fPC_LFM = np.multiply(np.fft.fft(echo[:, 0], n=Nfft), f_LFM.conjugate())
    PC_LFM = to_dB(np.absolute(np.fft.ifftshift(np.fft.ifft(fPC_LFM, n=Nfft))))
    fPC_SMSP = np.multiply(np.fft.fft(SMSP[:, 0], n=Nfft), f_LFM.conjugate())
    PC_SMSP = to_dB(np.absolute(np.fft.ifftshift(np.fft.ifft(fPC_SMSP, n=Nfft))))
    fPC_ISRJ = np.multiply(np.fft.fft(ISRJ[:, 0], n=Nfft), f_LFM.conjugate())
    PC_ISRJ = to_dB(np.absolute(np.fft.ifftshift(np.fft.ifft(fPC_ISRJ, n=Nfft))))
    fPC_INUSRJ = np.multiply(np.fft.fft(INUSRJ[:, 0], n=Nfft), f_LFM.conjugate())
    PC_INUSRJ = to_dB(np.absolute(np.fft.ifftshift(np.fft.ifft(fPC_INUSRJ, n=Nfft))))
    PCdata = np.concatenate((np.expand_dims(PC_LFM, axis=1), np.expand_dims(PC_SMSP, axis=1),
                             np.expand_dims(PC_ISRJ, axis=1), np.expand_dims(PC_INUSRJ, axis=1)), axis=1)
    np.savetxt('pc_data\PCdata.csv', PCdata, delimiter=',')

def velocity_disturbance(args):
    velocity_old = args.velocity
    velocity = args.velocity + args.v_sigma * np.random.rand()
    return velocity_old, velocity

def Get_sample(args, Data, type, data_num, miu_list, t_line, SNR, JSR):
    for i in tqdm(range(int(data_num))):
        # velocity_old, velocity = velocity_disturbance(args)
        # args.velocity = velocity
        args.miu = miu_list[i]
        # args.miu = random.choice((-1, 1)) * random.uniform(args.miu_min, args.miu_max)
        LFM = np.exp(1j * np.pi * args.miu * t_line * t_line)  # 线性调频信号 # echo = sqrt(math.pow(10, -SNR/10)) * noise + LFM
        tran_sig = generate_signal(LFM, args=args)
        echo = add_noise_to_signal(tran_sig, SNR[i])
        echo_j, _ = add_jamming_to_signal(echo, JSR=JSR[i], type=type, args=args)  # 添加干扰信号，满足JSR
        data = data_change(echo_j)
        LFM_data = data_change(tran_sig)
        Data.append(np.concatenate((data, LFM_data), axis=0))
        # scio.savemat('230526testdata.mat', {'echo_j': echo_j, 'tran_sig':tran_sig})
        # args.velocity = velocity_old
    return Data

if __name__ == '__main__':
    args = parameter_setting()
    args = args.parse_args()
    args.main_file = "Power_" + args.type
    # args.main_file = "Copy_Power_" + args.type
    args.output_dir = args.main_file + '\epochs\model'
    os.makedirs(args.output_dir, exist_ok=True)
    t_line = np.arange(-args.TpLen / 2, args.TpLen / 2) / args.Fs  # 时间轴
    # N_Tr = np.dot(1 / args.prf, np.arange(0, args.num_pluses, 1))
    # N_Theta = np.expand_dims(np.exp(1j * 2 * np.pi * N_Tr), 0)
    miu_list = np.linspace(args.miu_min, args.miu_max, int(args.test_data_number / 2))
    miu_list = np.append(np.flipud(-miu_list), miu_list)
    eva_path = args.main_file + "/Evaluate"
    os.makedirs(eva_path, exist_ok=True)
    args.eva_path = eva_path + '/data_eva.npz'

    print('正在保存设置参数...')
    path = args.main_file + '/setting/'
    os.makedirs(path, exist_ok=True)
    joblib.dump(args, path + 'parameter_setting.pkl')
    joblib.dump(args, 'parameter_setting.pkl')
    print('参数保存完毕！\n')

    type_list = args.type.split('-', 1)
    test_list = type_list[1].split('&')
    train_list = type_list[0].split('&')

    # PC_data(args=args)  # 生成干扰的脉压结果

    # Data = [t_line]
    # Data_c = [t_line]
    # args.miu = 3E+9
    # LFM = np.exp(1j * np.pi * args.miu * t_line * t_line)
    # echo = generate_signal(LFM, args=args)
    # label = data_change(echo, Theoretical=1)
    # Data.append(label[0, :, 0])
    # Data_c.append(label[0, :, 0] + 1j * label[1, :, 0])
    # for i, type in enumerate(['SMSP', 'ISRJ', 'INUSRJ']):
    #     _, jamming = add_jamming_to_signal(echo, JSR=args.JSR, type=type, args=args)
    #     jamming = data_change(jamming, Theoretical=1)
    #     Data.append(jamming[0, :, 0])
    #     Data_c.append(jamming[0, :, 0] + 1j * jamming[1, :, 0])
    # Data = np.array(Data).T
    # Data_c = np.array(Data_c).T
    # np.savetxt('Theoretical_Signals.csv', Data, delimiter=',')
    # scio.savemat('complex_Theoretical_Signals.mat', {'Data_c': Data_c})
    # # np.savetxt('complex_Theoretical_Signals.csv', Data_c, delimiter=',')
    # print('样本数据生成完毕！')

    print('正在生成评估数据...')
    eva_Data = []
    eva_data_num = 8
    eva_miu_list = random.sample(miu_list.tolist(), eva_data_num)
    eva_SNR = np.linspace(-20, -20, int(eva_data_num))
    eva_JSR = np.random.uniform(args.JSR[0], high=args.JSR[-1] + 1, size=int(eva_data_num))
    for type in train_list:
        eva_Data = Get_sample(args, eva_Data, type, eva_data_num, eva_miu_list, t_line, eva_SNR, eva_JSR)
    np.savez(args.eva_path, data=eva_Data)
    print('正在生成测试数据...')
    test_total_num = args.SNR.shape[0] * args.JSR.shape[0] * args.test_data_number
    train_total_num = args.train_data_number
    for index_SNR in range(int(args.SNR.shape[0])):
        test_SNR = [args.SNR[index_SNR]] * int(args.test_data_number)
        for index_JSR in range(int(args.JSR.shape[0])):
            print("正在生成 SNR:{} - JSR：{} 的数据".format(int(args.SNR[index_SNR]), int(args.JSR[index_JSR])))
            test_JSR = [args.JSR[index_JSR]] * int(args.test_data_number)
            test_Data = []
            for type in test_list:
                test_Data = Get_sample(args, test_Data, type, args.test_data_number, miu_list, t_line, test_SNR, test_JSR)
            path = args.main_file + ("/Data_set/SNR_%.0f" % int(args.SNR[index_SNR])) + (
                    "/JSR_%.0f/" % int(args.JSR[index_JSR]))
            os.makedirs(path, exist_ok=True)
            np.savez(path + 'data_test', data=test_Data)
    print('正在生成训练数据...')
    train_Data = []
    train_miu_list = []
    for i in tqdm(range(int(args.train_data_number)), desc="生成训练需要的调频率："):
        while True:
            miu = random.choice((-1, 1)) * random.uniform(args.miu_min, args.miu_max)
            if miu not in miu_list:
                train_miu_list.append(miu)
                break
    train_SNR = np.random.uniform(args.SNR[0], high=args.SNR[-1]+1, size=int(args.train_data_number))
    train_JSR = np.random.uniform(args.JSR[0], high=args.JSR[-1]+1, size=int(args.train_data_number))
    for type in train_list:
        train_Data = Get_sample(args, train_Data, type, args.train_data_number, train_miu_list, t_line, train_SNR, train_JSR)
    path = args.main_file + "/Data_set/"
    os.makedirs(path, exist_ok=True)
    np.savez(path + 'total_data_train', data=train_Data)
    print('\n 数据生成完毕！\n')


    # train_Data = []
    # train_list = type_list[0].split('&')
    # for type in train_list:
    #     for i in range(int(args.train_data_number)):
    #         # velocity_old, velocity = velocity_disturbance(args)
    #         # args.velocity = velocity
    #         train_total_index = train_total_index + 1
    #         print("\r", end="")
    #         print("测试数据({:.2f}%): {}/{}  ||  ".format(round(test_total_index / test_total_num * 100, 2),
    #                                                   int(test_total_index),
    #                                                   int(test_total_num)),
    #               "训练数据({:.2f}%): {}/{}".format(round(train_total_index / train_total_num * 100, 2),
    #                                             int(train_total_index),
    #                                             int(train_total_num)),
    #               end="")
    #         sys.stdout.flush()
    #         # args.miu = random.choice((-1, 1)) * random.uniform(args.miu_min, args.miu_max)
    #         while True:
    #             args.miu = random.choice((-1, 1)) * random.uniform(args.miu_min, args.miu_max)
    #             if args.miu not in miu_list:
    #                 break
    #         # JSR = random.randint(-10, 50)
    #         JSR = random.randint(5, 15)
    #         # SNR = random.randint(-60, 0)
    #         # JSR = random.randint(8, 12)
    #         SNR = 10
    #         LFM = np.exp(
    #             1j * np.pi * args.miu * t_line * t_line)  # 线性调频信号 # echo = math.pow(10, -SNR/10) * noise + LFM
    #         tran_sig = generate_signal(LFM, args=args)
    #         echo = add_noise_to_signal(tran_sig, SNR)
    #         echo_j, _ = add_jamming_to_signal(echo, JSR=JSR, type=type, args=args)  # 添加干扰信号，满足JSR
    #         train_data = data_change(echo_j)
    #         LFM_matrix = np.expand_dims(LFM, 1) * N_Theta
    #         train_LFM_data = data_change(LFM_matrix)
    #         train_Data.append(np.concatenate((train_data[0:2, :, :], train_LFM_data[0:2, :, :]), axis=0))
    #         # args.velocity = velocity_old
    # path = "Power_" + args.type + "/Data_set/"
    # os.makedirs(path, exist_ok=True)
    # np.savez(path + 'total_data_train', data=train_Data)
    # print('\n 数据生成完毕！\n')

    # 保存数据文件
    # print('测试数据生成完毕！\n')
    # Data = []
    # if args.type in ['SMSP', 'ISRJ', 'INUSRJ']:
    #     for i in range(int(args.train_data_number)):
    #         # args.miu = random.choice((-1, 1)) * random.uniform(args.miu_min, args.miu_max)
    #         while True:
    #             args.miu = random.choice((-1, 1)) * random.uniform(args.miu_min, args.miu_max)
    #             if args.miu not in miu_list:
    #                 break
    #         JSR = random.randint(-10, 100)
    #         SNR = random.randint(-100, 20)
    #         LFM = np.exp(1j * np.pi * args.miu * t_line * t_line)  # 线性调频信号 # echo = math.pow(10, -SNR/10) * noise + LFM
    #         tran_sig = generate_signal(LFM, args=args)
    #         echo = add_noise_to_signal(tran_sig, SNR)
    #         echo_j, _ = add_jamming_to_signal(echo, JSR=JSR, type=args.type, args=args)  # 添加干扰信号，满足JSR
    #         train_data = data_change(echo_j)
    #         LFM_matrix = np.expand_dims(LFM, 1) * N_Theta
    #         train_LFM_data = data_change(LFM_matrix)
    #         Data.append(np.concatenate((train_data, train_LFM_data), axis=0))
    #         total_index = total_index + 1
    #         print("\r", end="")
    #         print("测试数据({}%): {}/{}  ||  ".format(round(test_total_index / test_total_num * 100),
    #                                               int(test_total_index), int(test_total_num)),
    #               "训练数据({}%): {}/{}".format(round(train_total_index / train_total_num * 100),
    #                                         int(train_total_index), int(train_total_num)),
    #               end="")
    #         sys.stdout.flush()
    # else:
    #     train_list = type_list[0].split('&')
    #     print(train_list)
    #     for type in train_list:
    #         for i in range(int(args.train_data_number)):
    #             # args.miu = random.choice((-1, 1)) * random.uniform(args.miu_min, args.miu_max)
    #             while True:
    #                 args.miu = random.choice((-1, 1)) * random.uniform(args.miu_min, args.miu_max)
    #                 if args.miu not in miu_list:
    #                     break
    #             JSR = random.randint(-10, 100)
    #             SNR = random.randint(-100, 20)
    #             LFM = np.exp(1j * np.pi * args.miu * t_line * t_line)  # 线性调频信号 # echo = math.pow(10, -SNR/10) * noise + LFM
    #             tran_sig = generate_signal(LFM, args=args)
    #             echo = add_noise_to_signal(tran_sig, SNR)
    #             echo_j, _ = add_jamming_to_signal(echo, JSR=JSR, type=type, args=args)  # 添加干扰信号，满足JSR
    #             train_data = data_change(echo_j)
    #             LFM_matrix = np.expand_dims(LFM, 1) * N_Theta
    #             train_LFM_data = data_change(LFM_matrix)
    #             Data.append(np.concatenate((train_data, train_LFM_data), axis=0))
    #             total_index = total_index + 1
    #             print("\r", end="")
    #             print("测试数据({}%): {}/{}  ||  ".format(round(test_total_index / test_total_num * 100),
    #                                                   int(test_total_index), int(test_total_num)),
    #                   "训练数据({}%): {}/{}".format(round(train_total_index / train_total_num * 100),
    #                                             int(train_total_index), int(train_total_num)),
    #                   end="")
    #             sys.stdout.flush()
    # path = "Temporary_data"
    # os.makedirs(path, exist_ok=True)
    # np.savez(path + '/total_data_train', data=Data)
    # print('训练数据生成完毕！\n')

    # t_line = np.arange(-args.TpLen / 2, args.TpLen / 2) / args.Fs
    # plt.subplot(2, 1, 1)
    # plt.plot(t_line, train_data[0, :, 0])
    # plt.xlabel('time')
    # plt.ylabel('Am')
    # plt.title('jamming')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.subplot(2, 1, 2)
    # plt.plot(t_line, train_LFM_data[0, :, 0])
    # plt.xlabel('time')
    # plt.ylabel('Am')
    # plt.title('signal')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.show()
    # dd = 1
