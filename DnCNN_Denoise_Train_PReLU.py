import matplotlib

matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os, glob, re, datetime, time
import torch
from torch.nn import init
import torch.utils.data as Data
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.optim as optim
import Sample_maker_forDN as SM

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')  # 创建一个解析对象
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--train_data', default='Sample\OriginalData', type=str, help='path of train data')
parser.add_argument('--sigma', default=[0.05,0.2,0.005], type=list, help='noise level : float or list')
parser.add_argument('--epoch', default=400, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--trainType', default='Test_LR', type=str, help='train the model with specific type noise')
parser.add_argument('--Noise2Noise',default=False,type=bool,help='Noise2Noise or not(True or False)')
parser.add_argument('--lr_decay',default=500,type=int,help='set the learning rate decay after  how many epochs ')
args = parser.parse_args()  # 解析参数

batch_size = args.batch_size
n_epoch = args.epoch
sigma = args.sigma

if isinstance(sigma, float):
    save_dir = os.path.join('models', args.model + '_' + 'PReLU' + '_' + 'sigma' + str(sigma))
if isinstance(sigma, list):
    save_dir = os.path.join('models', args.model + '_' + 'PReLU' + '_' + args.trainType)

# if path is not exist ,creat it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# DnCNN created
class DnCNN(nn.Module):
    def __init__(self, depth=17, image_channels=1, n_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
        '''
        DnCNN----无池化层，with Batch Normalizition and with out ResNet
        :param depth: DnCNN 网络深度
        :param n_channels: 每层提取多少个特征，即有多少卷积核
        :param image_channels: 输入的图像的通道数（灰度图为1）
        :param kernel_size: 卷积核的size
        :param strid: 卷积核扫描步长
        :param padding: 边界处理，补零宽度
        :param bias: 偏置
        '''
        super(DnCNN, self).__init__()
        layers = []  # 存储卷积层层，深度由depth决定

        # 第一层不含BN的卷积层
        layers.append(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
        )
        layers.append(nn.PReLU())

        # 添加中间Conv+BN层
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))  # 添加BN层
            layers.append(nn.PReLU())

        # 最后一层无BN卷积层
        layers.append(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=image_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
        )
        self.dncnn = nn.Sequential(*layers)  # 拆解layers构建网络
        self._initialize_weights()  # 参数初始化

    ##预测值为噪音剖面，返回值为匹配减之后的denoise剖面
    def forward(self, x):
        '''
        :param self:
        :param x: 输入的含噪剖面
        :return
            residual-out: 压噪后的剖面
            out:噪音剖面
        '''
        residual = x
        out = self.dncnn(x)
        return residual - out, out

    # 参数初始化函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)  # 将正交矩阵放入weights
                print("init weigts:{}".format(m.weight.size()))
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def findLastCheckpoint(savedir):
    file_list = glob.glob(os.path.join(savedir, 'model_*.tar'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).tar.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


# Loss函数
class sum_square_error(_Loss):
    """
        Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
        The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_square_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(
            2)  # div_(2)=除以2


if __name__ == '__main__':
    # model selection
    print("Build model!")
    model = DnCNN(image_channels=1, n_channels=64, depth=17, kernel_size=3)
    print(model)

    # # loss_func and optimizer
    criterion = sum_square_error()  # loss_func
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer=optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)

    # # load parameters
    initial_epoch = 0
    try:
        initial_epoch = findLastCheckpoint(save_dir)  #
        # checkpoint=torch.load(os.path.join(save_dir,os.listdir(save_dir)[0])) #load
        checkpoint = torch.load(os.path.join(save_dir, 'model_{}.tar'.format(initial_epoch)))  # load
        initial_epoch = checkpoint['epoch']
        if initial_epoch > 0:
            model.load_state_dict(checkpoint['model_state_dict'])  # model 权重和偏置
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # convert optimizer's state_dict to gpu
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    except:
        print("There is no trained model here!")
    finally:
        print("model file checked!")

    # set train mode
    model.train()

    # check GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # put the model and data into GPU
    model = model.to(device)
    criterion = criterion.to(device)
    ###### learning rates scheduler ######
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 350,450], gamma=0.7,last_epoch=-1)
    # scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)
    # scheduler=optim.lr_scheduler.CyclicLR(optimizer,max_lr=0.001,base_lr=0.00001,step_size_up=2000)
    # scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=100,verbose=True,threshold=0.1,min_lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_decay, T_mult=1, eta_min=1e-6,
                                                               last_epoch=-1)
    #Test learning rate schedule
    Loss=[]
    for epoch in range(initial_epoch, n_epoch):
        print("epoch:{}".format(epoch + 1), '\n')
        # generate clean data set
        xs = SM.datagenerator(data_dir=args.train_data)
        # xs = xs.astype('float32')
        xs = torch.from_numpy(xs).float()  # tensor of the clean patches from numpy, NXCXHXW
        DDataset = SM.DenoisingDataset(xs, sigma,noise2noise=args.Noise2Noise)  # return batch_ys(include noise) batch_xs(clean) or noise2noise set
        DLoader = Data.DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        iters = len(DLoader)
        epoch_loss = 0
        start_time = time.time()

        for n_count, (batch_y, batch_x) in enumerate(DLoader):
            # put data into GPU
            batch_y = batch_y.to(device)
            batch_x = batch_x.to(device)
            # '''Sample example view'''
            # plt.figure()
            # plt.subplot(3, 3, 1)
            # plt.imshow(batch_y[0][0].cpu())
            # plt.subplot(3, 3, 2)
            # plt.imshow(batch_y[10][0].cpu())
            # plt.subplot(3, 3, 3)
            # plt.imshow(batch_y[20][0].cpu())
            # plt.subplot(3, 3, 4)
            # plt.imshow(batch_y[3][0].cpu())
            # plt.subplot(3, 3, 5)
            # plt.imshow(batch_y[30][0].cpu())
            # plt.subplot(3, 3, 6)
            # plt.imshow(batch_y[50][0].cpu())
            # plt.subplot(3, 3, 7)
            # plt.imshow(batch_y[16][0].cpu())
            # plt.subplot(3, 3, 8)
            # plt.imshow(batch_y[27][0].cpu())
            # plt.subplot(3, 3, 9)
            # plt.imshow(batch_y[8][0].cpu())
            # plt.show()
            '''Sample example view'''
            [image, noise] = model(batch_y)
            scheduler.step(epoch + n_count / iters)
            loss = criterion(image, batch_x)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)  # step to the learning rate in this epcoh
            # scheduler.step()
            if n_count % 10 == 0:
                Loss.append(loss.item()/batch_size)
                print('%4d %4d / %4d loss = %2.4f' % (
                    epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
                print('learning rate:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')

    # save model's parameters
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, os.path.join(save_dir, 'model_%03d.tar' % (epoch + 1)))

    #save loss
    Loss_='withScheduleLR.npy'
    np.save(Loss_,np.array(Loss))
    # pre-view
    plt.figure(1)
    plt.imshow(batch_x.cpu().detach().numpy()[0][0])
    plt.figure(2)
    plt.imshow(batch_y.cpu().detach().numpy()[0][0])
    plt.figure(3)
    plt.imshow(image.cpu().detach().numpy()[0][0])
    plt.figure(4)
    plt.imshow(noise.cpu().detach().numpy()[0][0])
    plt.figure(5)
    plt.imshow(batch_x.cpu().detach().numpy()[0][0] - image.cpu().detach().numpy()[0][0])
    plt.show()
