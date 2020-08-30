import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
# from pyplotz.pyplotz import PyplotZ
# from pyplotz.pyplotz import plt
import argparse
import os, time, datetime, glob, re, pickle
import sys, segypy

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import Sample_maker_forDN

'''parameters'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='Sample\Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set03'], help='directory of test dataset')
    parser.add_argument('--sigma', default=0.1, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_PReLU_sigma0.1'),
                        help='directory of the model')
    parser.add_argument('--model_name', default='model_001.tar', type=str, help='the model name')
    parser.add_argument('--result_dir', default='Results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    parser.add_argument('--scanWidth', default=40, type=int, help='width of the scanning window')
    parser.add_argument('--scanLength', default=40, type=int, help='length of the scanning window')
    parser.add_argument('--scanCoverpercent',default=0.25,type=float,help='The cover percent of scan widows ( 0 <= C < scanLength )')
    return parser.parse_args()

'''plot function'''
class figure_plot:
    def __init__(self, data_dict=None, color='BrBG', title=True, img=False, n=0, show=False):
        self.data_dict = data_dict
        self.color = color
        self.title = title
        self.img = img
        self.n = n
        self.show = show

    def image(self, Data, cmap='coolwarm', colorbar=True, SH={}, maxval=-1):
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
        plt.pcolor(x, t, Data, cmap=plt.get_cmap(cmap), vmin=-1 , vmax=1)
        if colorbar:
            plt.colorbar()
            plt.minorticks_on()
        plt.axis('normal')
        plt.xlabel('Trace number')
        if 'time' in SH:
            plt.ylabel('Time (ms)')
        else:
            plt.ylabel('Sample number')
        if 'filename' in SH:
            plt.title(SH['filename'])
        plt.gca().invert_yaxis()

        # plt.grid(True)
        # plt.show()

    def wiggle(self, Data, SH={}, maxval=-1, skipt=1, lwidth=.5, x=[], t=[], gain=1, type='VA', color='black',
               ntmax=1e+9):
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
            print('segy wiggle: maxval = %g' % maxval)

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

    def show_wiggle(self):
        '''
        show the wiggle picture of input dict data
        :param data_dict: input data dict include origunal data and result et.
        :param title: add pictures' titles or not
        :param n: file number
        :param color: set color map of image
        '''
        data_dict = self.data_dict
        color = self.color
        title = self.title
        img = self.img
        n = self.n
        show = self.show

        for k, v in data_dict.items():
            if isinstance(v, list):
                try:
                    data = v[n]
                    if img:
                        plt.figure()
                        plt.subplot(121)
                        self.wiggle(data)
                        if title:
                            plt.title(k, x=1.1, y=1.03, fontsize=17)
                        plt.subplot(122)
                        self.image(data, cmap=color)
                    else:
                        plt.figure()
                        self.wiggle(data)
                        if title:
                            plt.title(k, y=1.03, fontsize=17)
                except:
                    print("The list is not in the right shape")
        if show:
            plt.show()

#中值滤波
def Med_feature(data, neigh_size):
    """
    中值滤波函数
    :param data: 输入数据
    :param neigh_size: 邻域大小
    :return: 滤波结果
    """
    Data_size = np.shape(data)
    fit_num = int((neigh_size - 1) / 2)
    # 首先对数据进行边界处理（补零）
    Data_insert_row_top = np.insert(data, 0, np.zeros((fit_num, Data_size[1])), axis=0)  # 矩阵上方补零
    Data_insert_row_bottom = np.insert(Data_insert_row_top, np.shape(Data_insert_row_top)[0],
                                       np.zeros((fit_num, Data_size[1])),
                                       axis=0)  # 矩阵下方补零
    Data_insert_colum_left = np.insert(Data_insert_row_bottom, 0,
                                       np.zeros((fit_num, np.shape(Data_insert_row_bottom)[0])), axis=1)  # 矩阵左边补零
    Data_insert = np.insert(Data_insert_colum_left, np.shape(Data_insert_colum_left)[1],
                            np.zeros((fit_num, np.shape(Data_insert_colum_left)[0])), axis=1)  # 矩阵右边补零
    # Data_insert:边界补零后的数组

    med_feature=np.empty(Data_size)

    for i in range(Data_size[0]):  # 逐行扫描
        for j in range(Data_size[1]):
            Neighbourhood=np.empty([neigh_size,neigh_size])
            for a in range(-fit_num,fit_num+1,1):
                # print(Data_insert[i + fit_num + a, j+fit_num - fit_num:j + fit_num +fit_num+ 1])
                Neighbourhood[a+fit_num,:]=Data_insert[i + fit_num + a, j+fit_num - fit_num:j + fit_num +fit_num+ 1] #邻域选取

            Neighbourhood_list=Neighbourhood.flatten()
            Neighbourhood_list.sort() #由小到大排序
            med_feature[i,j]=Neighbourhood_list[int((neigh_size*neigh_size-1)/2)] #获取中值
    return med_feature

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def save_result(result_dict, path):
    path = path if path.find('.') != -1 else path + '.pkl'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result_dict)
    elif ext == '.npy':
        np.save(path, results_dict)
    else:
        with open(path, 'wb') as f:
            pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)
        print("Data Dumped!")

class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3, padding=1):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.PReLU())
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.PReLU())
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        '''
        return denoise profile and noise
        :param x: noise-data
        :return: denoise profile / noise
        '''
        y = x
        out = self.dncnn(x)
        return y - out, out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

def findLastCheckpoint(dir):
    file_list = glob.glob(os.path.join(dir, 'model_*.tar'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).tar.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


if __name__ == '__main__':

    args = parse_args()

    model = DnCNN()  # 创建网络模型对象

    #### load state_dict for model ####
    try:
        initial_epock = findLastCheckpoint(args.model_dir)
        checkpoint = torch.load(os.path.join(args.model_dir, 'model_{}.tar'.format(initial_epock)))  # load parameters
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print('There is no trained model!')
        sys.exit(0)
    finally:
        print('model checked!')

    # set mode
    model.eval()  # evaluation mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:
        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))

        # initialization
        # 峰值信噪比和平均结构相似性
        psnrs = []
        ssims = []
        seisDataList = []
        noiseDataList = []
        resultsList = []
        noiseList = []
        residualList = []

        i = 0
        # load test data
        for seisData in os.listdir(os.path.join(args.set_dir, set_cur)):
            if seisData.endswith(".txt"):
                test_data = np.loadtxt(os.path.join(args.set_dir, set_cur, seisData))
            if seisData.endswith(".npy"):
                test_data = np.load(os.path.join(args.set_dir, set_cur, seisData))
            if seisData.endswith("sgy"):
                [test_data, SH, STH] = segypy.readSegy(os.path.join(args.set_dir, set_cur, seisData), endian='>')
            test_data = np.array(test_data)
            print(np.shape(test_data))
            data_size = np.shape(test_data)

            # to save result and noise profile
            result = np.zeros(np.shape(test_data))
            noiseProfile = np.zeros(np.shape(test_data))

            # normalization
            Norm = Sample_maker_forDN.Normalization_Func(test_data)
            test_data = Norm.Normalization_perTrace()
            # test_data = Norm.standardization()
            seisDataList.append(test_data)

            # noise
            np.random.seed(seed=0)  # for reproducibility
            noise = np.random.normal(0, args.sigma, np.shape(test_data))
            dataWithNoise = test_data + noise  # Add Gaussian noise without clipping
            noiseDataList.append(dataWithNoise)
            dataWithNoise = torch.from_numpy(dataWithNoise).float()  # convert to pytorch tensor

            torch.cuda.synchronize()
            start_time = time.time()
            dataWithNoise = dataWithNoise.to(device)  # sent to GPU

            for i in range(0, data_size[0], args.scanLength-int(args.scanCoverpercent * args.scanLength)):
                for j in range(0, data_size[1], args.scanLength-int(args.scanCoverpercent * args.scanLength)):
                    if i + args.scanLength > data_size[0] and j + args.scanWidth <= data_size[1]:
                        subArray = dataWithNoise[-args.scanLength:, j:j + args.scanWidth]
                        subArray = torch.reshape(subArray, [1, 1, args.scanLength, args.scanWidth])
                        [result_, noise_] = model(subArray)  # inference
                        result_ = result_.cpu().detach().numpy()
                        result_ = np.reshape(result_, (args.scanLength, args.scanWidth))
                        noise_ = noise_.cpu().detach().numpy()
                        noise_ = np.reshape(noise_, (args.scanLength, args.scanWidth))
                        result[-args.scanLength:, j:j + args.scanWidth] = result_
                        noiseProfile[-args.scanLength:, j:j + args.scanWidth] = noise_
                    elif j + args.scanWidth > data_size[1] and i + args.scanLength <= data_size[0]:
                        subArray = dataWithNoise[i:i + args.scanLength, -args.scanWidth:]
                        subArray = torch.reshape(subArray, [1, 1, args.scanLength, args.scanWidth])
                        [result_, noise_] = model(subArray)  # inference
                        result_ = result_.cpu().detach().numpy()
                        result_ = np.reshape(result_, (args.scanLength, args.scanWidth))
                        noise_ = noise_.cpu().detach().numpy()
                        noise_ = np.reshape(noise_, (args.scanLength, args.scanWidth))
                        result[i:i + args.scanLength, -args.scanWidth:] = result_
                        noiseProfile[i:i + args.scanLength, -args.scanWidth:] = noise_
                    elif i + args.scanLength > data_size[0] and j + args.scanWidth > data_size[1]:
                        subArray = dataWithNoise[-args.scanLength:, -args.scanWidth:]
                        subArray = torch.reshape(subArray, [1, 1, args.scanLength, args.scanWidth])
                        [result_, noise_] = model(subArray)  # inference
                        result_ = result_.cpu().detach().numpy()
                        result_ = np.reshape(result_, (args.scanLength, args.scanWidth))
                        noise_ = noise_.cpu().detach().numpy()
                        noise_ = np.reshape(noise_, (args.scanLength, args.scanWidth))
                        result[-args.scanLength:, -args.scanWidth:] = result_
                        noiseProfile[-args.scanLength:, -args.scanWidth:] = noise_
                    else:
                        subArray = dataWithNoise[i:i + args.scanLength, j:j + args.scanWidth]
                        subArray = torch.reshape(subArray, [1, 1, args.scanLength, args.scanWidth])
                        [result_, noise_] = model(subArray)  # inference
                        result_ = result_.cpu().detach().numpy()
                        result_ = np.reshape(result_, (args.scanLength, args.scanWidth))
                        noise_ = noise_.cpu().detach().numpy()
                        noise_ = np.reshape(noise_, (args.scanLength, args.scanWidth))
                        result[i:i + args.scanLength, j:j + args.scanWidth] = result_
                        noiseProfile[i:i + args.scanLength, j:j + args.scanWidth] = noise_

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time

            resultsList.append(result)
            noiseList.append(noiseProfile)
            residual = test_data - result
            residualList.append(residual)
            print('%10s : %10s : %2.4f second' % (set_cur, seisData, elapsed_time))

            # # check PSRN And SSIM
            psnr_x_ = peak_signal_noise_ratio(image_test=result, image_true=test_data,
                                              data_range=2)  # data_range=(-1,1)=1-(-1)=2
            ssim_x_ = structural_similarity(im1=test_data, im2=result)
            psnr_n_ = peak_signal_noise_ratio(image_test=dataWithNoise.cpu().detach().numpy(),image_true=test_data,data_range=2)
            ssim_n_ = structural_similarity(im1=test_data,im2=dataWithNoise.cpu().detach().numpy())
            psnrs.append(psnr_x_)
            ssims.append(ssim_x_)
            print("damaged data ------- signal_noise_ratio:{} ------- structural_similarity:{}".format(psnr_n_, ssim_n_))
            print('result ------- signal_noise_ratio:{} ------- structural_similarity:{}'.format(psnr_x_, ssim_x_))

        psnrs_average = np.mean(psnrs)
        ssims_average = np.mean(ssims)

        # save multi-data in a dict
        results_dict = {}
        results_dict['originals'] = seisDataList
        results_dict['noiseProfiles'] = noiseDataList
        results_dict['results'] = resultsList
        results_dict['noises'] = noiseList
        results_dict['residuals'] = residualList
        results_dict['psnrs'] = psnrs
        results_dict['ssims'] = ssims
        results_dict['psnrs_average'] = psnrs_average
        results_dict['ssims_average'] = ssims_average

        if args.save_result:
            if args.model_dir == "models\DnCNN_PReLU_Blind":
                file_name= "DnCNN_test_{}_sigma{}_Blind.pkl".format(args.set_names[0],args.sigma)
            elif args.model_dir == "models\DnCNN_PReLU_Noise2Noise":
                file_name = "DnCNN_test_{}_sigma{}_Noise2Noise.pkl".format(args.set_names[0],args.sigma)
            else:
                file_name = "DnCNN_test_{}_sigma{}.pkl".format(args.set_names[0],args.sigma)
            results_path = os.path.join(args.result_dir, set_cur, file_name)

            # # save the dict
            save_result(results_dict, results_path)
            log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnrs_average,
                                                                                ssims_average))  # # print imfo

            # # plot figure
            # ##enable chinese character
            # pltz=PyplotZ() # create an instance
            # pltz.enable_chinese() # enable chinese so can use chinese characters

            result_plot = figure_plot(data_dict=results_dict, color='seismic', title=True, img=True)
            result_plot.show_wiggle()
            # show_wiggle(results_dict,color='BrBG', img=False, n=0)

            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            # compare
            figure_plot = figure_plot()
            plt.figure()
            plt.subplot(131)
            figure_plot.image(results_dict['originals'][0], colorbar=False, cmap='gray')
            plt.tick_params(labelsize=13)
            plt.title('原始数据', y=1.02, fontsize=17)
            plt.subplot(132)
            figure_plot.image(results_dict['noiseProfiles'][0], colorbar=False, cmap='gray')
            plt.tick_params(labelsize=13)
            plt.title('加噪数据', y=1.02, fontsize=17)
            plt.subplot(133)
            figure_plot.image(results_dict['results'][0], cmap='gray',colorbar=False)
            plt.tick_params(labelsize=13)
            plt.title('压噪结果', y=1.02, fontsize=17)
            plt.show()
