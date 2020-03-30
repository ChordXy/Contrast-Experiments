'''
@Author: Cabrite
@Date: 2020-03-28 16:38:00
@LastEditors: Cabrite
@LastEditTime: 2020-03-30 16:44:50
@Description: Do not edit
'''

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import Load_MNIST
import datetime
import sklearn
import math
import gc
import os


#- 获取Gabor滤波器
def getGaborFilter(ksize, sigma, theta, lambd, gamma, psi, RI_Part = 'r', ktype = np.float64):
    """ This is my own design of Gabor Filters.
        The efficiency of this generator isn't good, waiting to be realized on C++ and generated as an API
    
    Arguments:
        ksize {tuple} -- The size of the kernel
        sigma {float} -- Sigma
        theta {float} -- Theta
        lambd {float} -- lambda
        gamma {float} -- Gamma
        psi {float} -- Psi
        ktype {int} -- np.float32 / np.float64
        RI_Part {char} -- Selete whether return real('r') , image part('i'), both ('b'). 
    """
    sigma_x = sigma
    sigma_y = sigma / gamma

    nstds = 3
    c = np.cos(theta)
    s = np.sin(theta)
    if (ksize[1] > 0):
        xmax = ksize[1] // 2
    else:
        xmax = np.round(np.max(np.fabs(nstds * sigma_x * c), np.fabs(nstds * sigma_y * s)))

    if (ksize[0] > 0):
        ymax = ksize[0] // 2
    else:
        ymax = np.round(np.max(np.fabs(nstds * sigma_x * s), np.fabs(nstds * sigma_y * c)))

    xmin = - xmax
    ymin = - ymax

    kernel = np.ones((ymax - ymin + 1, xmax - xmin + 1), dtype = ktype)

    scale = 1
    ex = -0.5 / (sigma_x * sigma_x)
    ey = -0.5 / (sigma_y * sigma_y)
    cscale = np.pi * 2 / lambd

    mesh_x, mesh_y = np.meshgrid(range(xmin, xmax + 1), range(ymin, ymax + 1))
    mesh_xr = mesh_x * c + mesh_y * s
    mesh_yr = - mesh_x * s + mesh_y * c
    GauPart = scale * np.exp(ex * mesh_xr * mesh_xr + ey * mesh_yr * mesh_yr)

    if RI_Part == 'r':
        # v_real = GauPart * np.cos(cscale * mesh_xr + psi)
        return GauPart * np.cos(cscale * mesh_xr + psi)
    elif RI_Part == 'i':
        # v_image = GauPart * np.sin(cscale * mesh_xr + psi)
        return GauPart * np.sin(cscale * mesh_xr + psi)
    else:
        return GauPart * np.cos(cscale * mesh_xr + psi), GauPart * np.sin(cscale * mesh_xr + psi)

#- Gabor网络
class GaborFeature():
    def __init__(self, ksize, Theta, Lambda, Gamma, Beta, RI_Part = 'r', ktype = np.float64, pool_result_size = 2):
        """初始化Gabor类
        """
        #- 初始化参数
        self.KernelSize = None
        self.ReturnPart = None
        self.KernelType = None
        self.Train_BN = None
        self.Test_BN = None
        self.__Gabor_params = []
        self.__Gabor_filter = None
        self.pool_size = pool_result_size

        self.setParam(ksize, Theta, Lambda, Gamma, Beta, RI_Part, ktype)
        self.getMNIST()
        self.generateGaborFeature()

    #- 设置Gabor参数
    def setParam(self, ksize, Theta, Lambda, Gamma, Beta, RI_Part = 'r', ktype = np.float64):
        """带BaudWidth参数的Gabor参数，默认  Psi = 0
        
        Arguments:
            ksize {tuple} -- Gabor核心
            Theta {list(float64)} -- 角度参数列表
            Lambda {list(float64)} -- 尺度参数列表
            Gamma {list(float64)} -- 横纵比参数列表
            Beta {list(float64)} -- 带宽参数列表
        
        Keyword Arguments:
            RI_Part {str} -- 返回虚部（'i'）还是实部（'r'）或者全部返回（'b'） (default: {'r'})
            ktype {in} -- 返回的Gabor核数据类型 (default: {np.float64})
        """
        self.KernelSize = ksize
        self.ReturnPart = RI_Part
        self.KernelType = ktype

        #* 将参数解包成列表，保存
        temp_res = []

        for lam in Lambda:
            for the in Theta:
                for gam in Gamma:
                    for bd in Beta:
                        temp_res.append([lam, the, gam, bd])

        #* beta转换成sigma，生成标准的参数集 [sigma, theta, lambda, gamma, psi]
        for lam, the, gam, bd in temp_res:
            lam = pow(2, 0.5 * (1 + lam))
            sig = 1 / np.pi * np.sqrt(np.log(2) / 2) * (pow(2, bd) + 1) / (pow(2, bd) - 1) * lam
            self.__Gabor_params.append([sig, the, lam, gam, 0])
        
        self.GenerateGaborFilter()

    #- 生成Gabor滤波器
    def GenerateGaborFilter(self):
        """生成Gabor滤波器组
        """
        self.__Gabor_filter = np.zeros([self.numGaborFilters, *self.KernelSize])
        index = 0
        
        self.PrintLog("Generating Gabor Filters...")

        if self.ReturnPart == 'b':
            for sig, the, lam, gam, ps in self.__Gabor_params:
                self.__Gabor_filter[index], self.__Gabor_filter[index + 1] = getGaborFilter(self.KernelSize, sig, the, lam, gam, ps, self.ReturnPart)
                index += 2
        else:
            for sig, the, lam, gam, ps in self.__Gabor_params:
                self.__Gabor_filter[index] = getGaborFilter(self.KernelSize, sig, the, lam, gam, ps, self.ReturnPart)
                index += 1

        #- 将Gabor滤波器翻转180°，并调整形状，以送入TensorFlow中进行卷积
        #@ 翻转180°
        for i in range(self.numGaborFilters):
            self.__Gabor_filter[i] = np.rot90(self.__Gabor_filter[i], 2)
        #@ 如果单通道，则需要新增一根轴，表明是单通道；如果多通道，则shape=4
        if len(self.__Gabor_filter.shape) == 3:
            self.__Gabor_filter = self.__Gabor_filter[:, :, :, np.newaxis]
        #@ 将滤波器个数的轴换到最后，适配TensorFlow中卷积滤波器的格式
        self.__Gabor_filter = self.__Gabor_filter.transpose(1, 2, 3, 0)

        self.PrintLog("Generating Gabor Filters Done!")

    #- 提取Gabor特征
    def ExtractGaborFeature(self, Images, batchsize=500, method='SAME'):
        """利用生成的Gabor滤波器组对图像进行卷积
        
        Arguments:
            Images {np.array[numImages, rows, cols]} -- 图像

        Keyword Arguments:
            method {str} -- 卷积方法 (default: {True})
        
        Returns:
            np.array[numFilter, imageSize] -- 返回滤波后的图像组
        """
        #- 图像数据预处理
        #@ 如果图像是单通道数据，则添加一根轴，表明单通道；多通道则不需要。适配TensorFlow卷积中图像的格式
        if len(Images.shape) == 3:
            Images = Images[:, :, :, np.newaxis]

        #- 初始化参数
        numImages, rows, cols, channel = Images.shape
        Krows, Kcols, kChannel, numKernels = self.__Gabor_filter.shape
        totalbatch = math.ceil(numImages / batchsize)
        result = None

        #- 定义网络
        tf.reset_default_graph()
        input_image = tf.placeholder(tf.float32, [None, rows, cols, channel])
        input_filter = tf.placeholder(tf.float32, [Krows, Kcols, kChannel, numKernels])

        conv = tf.nn.conv2d(input_image, input_filter, [1, 1, 1, 1], method)
        maxpool = tf.nn.max_pool(conv, [1, 28 // self.pool_size, 28 // self.pool_size, 1], [1, 28 // self.pool_size, 28 // self.pool_size, 1], 'VALID')
        reshaped_maxpool = tf.reshape(maxpool, [batchsize, self.pool_size * self.pool_size * 128])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tsg = self.PrintLog("Extracting Features...")
    
            for i in range(totalbatch):
                self.PrintLog("Extracting Features... {}/{}".format(min(i * batchsize, numImages), numImages))

                Selected_Images = Images[i * batchsize : min((i + 1) * batchsize, numImages)]
                result_pool = sess.run(reshaped_maxpool, feed_dict={input_image:Selected_Images, input_filter:self.__Gabor_filter})
                if i == 0:
                    result = result_pool
                else:
                    result = np.concatenate((result, result_pool), axis=0)
    
            self.PrintLog("Extracting Features Done!", tsg)

        return result

    #- 筛选Gabor特征
    def FeatureReduction(self, Train_X_feature, Test_X_feature):
        """特征缩减
        
        Arguments:
            Train_X_feature {np.array[numImages, Features]} -- 训练特征
            Test_X_feature {np.array[numImages, Features]} -- 测试特征
        
        Returns:
            np.array[numImages, SelectedFeatures], np.array[numImages, SelectedFeatures] -- 缩减后的特征
        """
        #* 计算标准差
        train_std = np.std(Train_X_feature, 0)
        #* 找到方差大于 10^-3 的位置
        Feature_Position = np.where(train_std > 1e-3)
        #* 提取出符合条件的方差
        train_std_Keep = train_std[Feature_Position]
        #* 提取出符合条件的训练及测试特征
        train_Keep = Train_X_feature[:, Feature_Position]
        test_Keep = Test_X_feature[:, Feature_Position]
        #* 归一化
        train_Keep_mean = np.mean(train_Keep, 0)
        train_Keep_Normalized = (train_Keep - train_Keep_mean) / train_std_Keep
        test_Keep_Normalized = (test_Keep - train_Keep_mean) / train_std_Keep
        #* 在裁片时，会多一根轴在中央，需要剔除
        train_Keep_Normalized = np.reshape(train_Keep_Normalized, [train_Keep_Normalized.shape[0], train_Keep_Normalized.shape[1] *  train_Keep_Normalized.shape[2]])
        test_Keep_Normalized = np.reshape(test_Keep_Normalized, [test_Keep_Normalized.shape[0], test_Keep_Normalized.shape[1] * test_Keep_Normalized.shape[2]])
        return train_Keep_Normalized, test_Keep_Normalized

    #- Gabor网络特征
    def generateGaborFeature(self):
        Train_Feature = self.ExtractGaborFeature(self.Train_X[0:1000])
        Test_Feature = self.ExtractGaborFeature(self.Test_X[0:1000])
        self.Train_BN, self.Test_BN = self.FeatureReduction(Train_Feature, Test_Feature)
        del Train_Feature, Test_Feature
        gc.collect()

    def getMNIST(self):
        """ 获取MNIST图像
        """
        self.Train_X, self.Train_Y, self.Test_X, self.Test_Y = Load_MNIST.Preprocess_MNIST_Data("./Datasets/MNIST_Data", True, True)

    @property
    def GaborTrainFeature(self):
        return self.Train_BN

    @property
    def GaborTestFeature(self):
        return self.Test_BN

    @property
    def numGaborFilters(self):
        """Gabor滤波器组中滤波器个数
        
        Returns:
            int -- 滤波器个数
        """
        numKernel = len(self.__Gabor_params)
        if self.ReturnPart == 'b':
            numKernel *= 2
        return numKernel

    #- 附加函数
    def PrintLog(self, message, diff_logTime=None):
        """打印Log信息
        
        Arguments:
            message {str} -- 要显示的信息
        
        Keyword Arguments:
            diff_logTime {datetime.datetime} -- 需要计算时间差的变量 (default: {None})
        
        Returns:
            datetime.datetime -- 当前的时间信息
        """
        nowTime = datetime.datetime.now()
        msg = "[" + nowTime.strftime('%Y-%m-%d %H:%M:%S.%f') + "] " + message
        print(msg)

        if isinstance(diff_logTime, datetime.datetime):
            diff_time = str((nowTime - diff_logTime).total_seconds())
            msg = "[" + nowTime.strftime('%Y-%m-%d %H:%M:%S.%f') + "] Time consumption : " + diff_time + ' s'
            print(msg)
        
        return nowTime

class AEFeature():
    def __init__(self):
        pass

def ClassifierSVM(train_x, train_y, test_x, test_y, kernel = 'rbf'):
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf').fit(train_x, train_y)

def ClassifierMLP():
    pass

def DimensionReduction(data, Targeted_Dimension = 2, method = 0):
    if method == 0:
        Reduced_Data = PCA(n_components=Targeted_Dimension).fit_transform(data)
    else:
        Reduced_Data = TSNE(n_components=Targeted_Dimension).fit_transform(data)
    return Reduced_Data

def ClassifierKMeans(data, n_class = 10):
    estimator = KMeans(n_clusters=n_class)
    estimator.fit(data)


#- 测试函数
def TestCluster():
    from mpl_toolkits.mplot3d import Axes3D

    n_samples = 1000
    data_1 = 10 * np.ones([n_samples, 3]) + np.random.uniform(0, 1, [n_samples, 3])
    data_2 = -10 * np.ones([n_samples, 3]) + np.random.uniform(0, 2, [n_samples, 3])
    data_3 = 30 * np.ones([n_samples, 3]) + np.random.uniform(0, 1.5, [n_samples, 3])
    data_4 = 50 * np.ones([n_samples, 3]) + np.random.uniform(0, 2.5, [n_samples, 3])
    data = np.concatenate([data_1, data_2, data_3, data_4])

    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o')

    X_Reduced = DimensionReduction(data, 2, 1)
    fig = plt.figure(2)
    plt.scatter(X_Reduced[:, 0], X_Reduced[:, 1], marker='o')
    plt.show()


if __name__ == "__main__":
    ksize = (11, 11)
    Lambda = [1, 2, 3, 4]
    numTheta = 8
    Theta = [np.pi / numTheta * i for i in range(numTheta)]
    Beta = [1]
    Gamma = [0.5, 1]

    # GaborFeatures = GaborFeature(ksize, Theta, Lambda, Gamma, Beta, 'b')

    
    TestCluster()