import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import segypy
import re
import glob
import os
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch


# plot func
def wiggle(Data, SH={}, maxval=-1, skipt=1, lwidth=.5, x=[], t=[], gain=1, type='VA', color='black', ntmax=1e+9):
    """
    wiggle(Data,SH)
    """
    import matplotlib.pylab as plt
    import numpy as np
    import copy

    yl = 'Sample number'

    ns = Data.shape[0]
    ntraces = Data.shape[1]

    if ntmax < ntraces:
        skipt = int(np.floor(ntraces / ntmax))
        if skipt < 1:
            skipt = 1

    if len(x) == 0:
        x = range(0, ntraces)

    if len(t) == 0:
        t = range(0, ns)
    else:
        yl = 'Time  [s]'

    # overrule time form SegyHeader
    if 'time' in SH:
        t = SH['time']
        yl = 'Time  [s]'

    dx = x[1] - x[0]
    if (maxval <= 0):
        Dmax = np.nanmax(Data)
        maxval = -1 * maxval * Dmax
        print('segypy.wiggle: maxval = %g' % maxval)

    # fig, (ax1) = plt.subplots(1, 1)'
    fig = plt.gcf()
    ax1 = plt.gca()

    for i in range(0, ntraces, skipt):
        # use copy to avoid truncating the data
        trace = copy.copy(Data[:, i])
        trace[0] = 0
        trace[-1] = 0
        ax1.plot(x[i] + gain * skipt * dx * trace / maxval, t, color=color, linewidth=lwidth)

        if type == 'VA':
            for a in range(len(trace)):
                if (trace[a] < 0):
                    trace[a] = 0;
                    # pylab.fill(i+Data[:,i]/maxval,t,color='k',facecolor='g')
            # ax1.fill(x[i] + dx * Data[:, i] / maxval, t, 'k', linewidth=0, color=color)
            ax1.fill(x[i] + gain * skipt * dx * trace / (maxval), t, 'k', linewidth=0, color=color)

    ax1.grid(True)
    ax1.invert_yaxis()
    plt.ylim([np.max(t), np.min(t)])

    plt.xlabel('Trace number')
    plt.ylabel(yl)
    if 'filename' in SH:
        plt.title(SH['filename'])
    # ax1.set_xlim(-1, ntraces)
    # plt.show()

def image(Data, SH={}, maxval=-1):
    """
    image(Data,SH,maxval)
    Image segy Data
    """
    import matplotlib.pylab as plt

    if (maxval <= 0):
        Dmax = np.max(Data)
        maxval = -1 * maxval * Dmax

    if 'time' in SH:
        t = SH['time']
        ntraces = SH['ntraces']
        ns = SH['ns']
    else:
        ns = Data.shape[0]
        t = np.arange(ns)
        ntraces = Data.shape[1]
    x = np.arange(ntraces) + 1

    print(maxval)
    plt.pcolor(x, t, Data, vmin=-1 * maxval, vmax=maxval)
    plt.colorbar()
    plt.axis('normal')
    plt.xlabel('Trace number')
    if 'time' in SH:
        plt.ylabel('Time (ms)')
    else:
        plt.ylabel('Sample number')
    if 'filename' in SH:
        plt.title(SH['filename'])
    plt.gca().invert_yaxis()


''' 数据预处理——归一化 '''
class Normalization_Func():
    def __init__(self, input_data):
        self.input_data = input_data

    def standardization(self):
        mu = np.mean(self.input_data, axis=0)
        sigma = np.std(self.input_data, axis=0)
        return (self.input_data - mu) / sigma

    def Normalization_perTrace(self):
        Data_size = np.shape(self.input_data)
        Norm_data = np.empty(Data_size)
        for j in range(Data_size[1]):
            Data_max = np.max(np.abs(self.input_data[:, j]))
            Norm_data[:, j] = self.input_data[:, j] / Data_max
        return Norm_data

    def Normalization_perTrace_2(self):
        Data_size = np.shape(self.input_data)
        Norm_data = np.empty(Data_size)
        for j in range(Data_size[1]):
            Data_max = np.max(self.input_data[:, j])
            Data_min = np.min(self.input_data[:, j])
            Norm_data[:, j] = (self.input_data[:, j] - Data_min) / (Data_max - Data_min)
        return Norm_data

    def Normalization_MinMax_Negative(self):
        Xmax = np.max(self.input_data)
        Xmin = np.min(self.input_data)
        Norm_input_data = 2 * (self.input_data - Xmin) / (Xmax - Xmin) - 1
        return Norm_input_data

    def Normalization(self):  # 均值0 方差1
        from numpy.matlib import repmat
        rows, cols = np.shape(self.input_data)
        if rows == 1:
            input_data = np.transpose(self.input_data)
            len = cols
            num = 1
        else:
            len = rows
            num = cols
        MeanChaos = np.mean(input_data)
        input_data = self.input_data - repmat(MeanChaos, len, 1)  # 0均值
        w = 1 / np.std(input_data, ddof=1)
        Output = np.multiply(input_data, repmat(w, len, 1))  # 方差1
        return Output

    def Z_ScoreNormalization(self):
        mu = np.average(self.input_data)
        Sigma = np.std(self.input_data)
        Norm_Input_data = (self.input_data - mu) / Sigma
        return Norm_Input_data


###########################################################################

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 0.25
    """

    def __init__(self, xs, sigma,noise2noise=False):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigmaList=sigma
        self.noise2noise=noise2noise

    def __getitem__(self, index):
        batch_x = self.xs[index]
        batch_x_tmp=batch_x
        if isinstance(self.sigmaList,float):
            GaussianNoise = np.random.normal(0, self.sigmaList,
                                             (np.shape(self.xs[0][0])[0], np.shape(self.xs[0][0])[1]))  # 高斯白噪声
            if self.noise2noise:
                batch_x_tmp=batch_x+torch.from_numpy(np.random.normal(0,self.sigmaList,(np.shape(self.xs[0][0])[0], np.shape(self.xs[0][0])[1]))).float()

        if isinstance(self.sigmaList,list):
            sigma=np.random.choice(np.arange(self.sigmaList[0],self.sigmaList[1],self.sigmaList[2]))  #噪声强度
            GaussianNoise = np.random.normal(0,sigma ,
                                             (np.shape(self.xs[0][0])[0], np.shape(self.xs[0][0])[1]))  # 高斯白噪声
            if self.noise2noise:
                batch_x_tmp=batch_x+torch.from_numpy(np.random.normal(0,sigma,(np.shape(self.xs[0][0])[0], np.shape(self.xs[0][0])[1]))).float()

        GaussianNoise = torch.from_numpy(GaussianNoise).float()
        batch_y = batch_x + GaussianNoise
        return batch_y, batch_x_tmp

    def __len__(self):
        return self.xs.size(0)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)  # shrink the image
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def gen_patches(file_name):
    pass


def getData(dataDir):
    '''
    read segy npy txt
    :param dataDir: file path
    :return: data set
    '''
    fileList = os.listdir(dataDir)  # 获取目标文件夹所有文件名的列表
    Profile = []
    for se in fileList:
        if se.endswith(".npy"):
            path = os.path.join(dataDir, se)
            data = np.load(path)
            Profile.append(data)
        if se.endswith(".sgy"):
            path = os.path.join(dataDir, se)
            # Read Segy File
            segypy.verbose = 1
            [data, SH, STH] = segypy.readSegy(path, endian='>')
            Profile.append(data)
        if se.endswith(".txt"):
            path = os.path.join(dataDir, se)
            data = np.loadtxt(path)
            Profile.append(data)
    print("_____________Data loaded_____________")
    return Profile


# split input array into subarray
def dataSplit(profile, kernel_size=40):
    '''
    split input array into subarray
    :param profile: 输入的数组
    :param kernel_size: 子数组大小
    :return subArrays:子数组列表
    '''
    data_size = np.shape(profile)
    subArrays = []
    for i in range(0, data_size[0], kernel_size):
        for j in range(0, data_size[1], kernel_size):
            if i + kernel_size > data_size[0] and j + kernel_size <= data_size[1]:
                subArray = profile[-kernel_size :, j:j + kernel_size]
                subArrays.append(subArray)
            elif j + kernel_size > data_size[1] and i + kernel_size <= data_size[0]:
                subArray = profile[i:i + kernel_size, -kernel_size :]
                subArrays.append(subArray)
            elif i + kernel_size > data_size[0] and j + kernel_size > data_size[1]:
                subArray = profile[-kernel_size :, -kernel_size :]
                subArrays.append(subArray)
            else:
                subArray = profile[i:i + kernel_size, j:j + kernel_size]
                subArrays.append(subArray)
    return subArrays


def datagenerator(data_dir='Sample\OriginalData', norm=True, verbose=False):
    # generate clean patches from a dataset
    Profiles = getData(data_dir)  # loaded profiles
    # Normalization
    if norm:
        for i in range(len(Profiles)):
            Norm = Normalization_Func(Profiles[i])
            Profiles[i] = Norm.Normalization_perTrace()
    # initrialize
    data = []
    # generate patches
    for i in range(len(Profiles)):
        patches = dataSplit(Profiles[i], kernel_size=40)
        for patch in patches:
            patch = np.reshape(patch, (1, np.shape(patch)[0], np.shape(patch)[1]))
            data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(Profiles)) + ' is done ^_^')
    data = np.array(data)
    print("样本大小{}".format(np.shape(data)))
    # discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch normalization
    # data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data


if __name__ == '__main__':
    data = datagenerator(data_dir='Sample\OriginalData')
    print(type(data))

    #    print('Shape of result = ' + str(res.shape))
    #    print('Saving data...')
    #    if not os.path.exists(save_dir):
    #            os.mkdir(save_dir)
    #    np.save(save_dir+'clean_patches.npy', res)
    #    print('Done.')

    # #载入原始数据
    # Profile_ori=np.load('D:\code\python\project\Machine-learning\CNN_Pytorch_SW_Denoise\Sample\OriginalData\OriginalData.npy')
    # Profile_size=np.shape(Profile_ori)
    # print("剖面大小：",Profile_size)
    #
    # Data=getData('D:\code\python\project\Machine-learning\CNN_Pytorch_SW_Denoise\Sample\OriginalData')
    # print(len(Data))
    #
    # #噪音剖面生成
    # Noise=np.random.normal(0,0.2,np.shape(Profile_ori))
    # #数据归一化
    # norm=Normalization_Func(Profile_ori)
    # Profile_norm=norm.Normalization_perTrace()
    #
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(data[0][0])
    # plt.subplot(222)
    # plt.imshow(data[1][0])
    # plt.subplot(223)
    # plt.imshow(data[200][0])
    # plt.subplot(224)
    # plt.imshow(data[300][0])
    # plt.figure()
    # wiggle(Profile_ori)
    # plt.figure()
    # wiggle(Profile_norm)
    # plt.figure()
    # wiggle(Noise+Profile_norm)
    # plt.figure()
    # image(Noise+Profile_norm)
    # plt.show()
