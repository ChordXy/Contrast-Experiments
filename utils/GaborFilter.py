'''
@Author: Cabrite
@Date: 2020-07-05 09:45:19
@LastEditors: Cabrite
@LastEditTime: 2020-07-17 21:57:24
@Description: Gabor 濾波器
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import Loggers
import math


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

#- Gabor类
class Gabor():
    def __init__(self):
        """初始化Gabor类
        """

        #- 初始化参数
        self.KernelSize = None
        self.ReturnPart = None
        self.KernelType = None
        self.__Gabor_params = []
        self.__Gabor_filter = None
        self.__Gabor_filter_size = []
        self.__Gabor_filter_area = []
        self.__Gabor_filter_name = []

    #- 设置Gabor参数
    def setParam(self, ksize, Sigma, Theta, Lambda, Gamma, Psi = [0], RI_Part = 'r', ktype = np.float64):
        """标准Gabor参数，标准格式：[sigma, theta, lambda, gamma, psi]
        
        Arguments:
            ksize {tuple} -- Gabor核心
            Sigma {list(float64)} -- 缩放比参数列表
            Theta {list(float64)} -- 角度参数列表
            Lambda {list(float64)} -- 尺度参数列表
            Gamma {list(float64)} -- 横纵比参数列表
        
        Keyword Arguments:
            Psi {list(float64)} -- Psi值列表 (default: {[0]})
            RI_Part {str} -- 返回虚部（'i'）还是实部（'r'）或者全部返回（'b'） (default: {'r'})
            ktype {in} -- 返回的Gabor核数据类型 (default: {np.float64})
        """
        self.KernelSize = ksize
        self.ReturnPart = RI_Part
        self.KernelType = ktype

        #* 将参数解包成列表，保存
        for sig in Sigma:
            for the in Theta:
                for lam in Lambda:
                    for gam in Gamma:
                        for ps in Psi:
                            self.__Gabor_params.append([sig, the, lam, gam, ps])

        self.GenerateGaborFilter()
        
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
            self.__Gabor_filter_name.append("σ={} θ={} λ={} γ={}".format(sig, the, lam, gam))
        
        self.GenerateGaborFilter()

    #- 生成Gabor滤波器
    def GenerateGaborFilter(self):
        """生成Gabor滤波器组
        """
        self.__Gabor_filter = np.zeros([self.numGaborFilters, *self.KernelSize])
        index = 0
        
        Loggers.TFprint.TFprint("Generating Gabor Filters...")

        if self.ReturnPart == 'b':
            for sig, the, lam, gam, ps in self.__Gabor_params:
                # 倒金字塔结构
                gabor_pat_size = 3 - int(np.round((sig / 2 - 0.5) * 3))
                gabor_pat_area = pow(2 * gabor_pat_size + 1, 2)
                self.__Gabor_filter_size.append(gabor_pat_size)
                self.__Gabor_filter_size.append(gabor_pat_size)
                self.__Gabor_filter_area.append(gabor_pat_area)
                self.__Gabor_filter_area.append(gabor_pat_area)
                self.__Gabor_filter[index], self.__Gabor_filter[index + 1] = getGaborFilter(self.KernelSize, sig, the, lam, gam, ps, self.ReturnPart)
                index += 2
        else:
            for sig, the, lam, gam, ps in self.__Gabor_params:
                # 倒金字塔结构
                gabor_pat_size = 3 - int(np.round((sig / 2 - 0.5) * 3))
                gabor_pat_area = pow(2 * gabor_pat_size + 1, 2)
                self.__Gabor_filter_size.append(gabor_pat_size)
                self.__Gabor_filter_area.append(gabor_pat_area)
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

        Loggers.TFprint.TFprint("Generating Gabor Filters Done!")

    #- 利用生成的Gabor滤波器卷积图像
    def ConvoluteImages(self, Images, batchsize=5000, method='SAME', isLogInfo=True):
        """利用生成的Gabor滤波器组对图像进行卷积
        
        Arguments:
            Images {np.array[numImages, rows, cols]} -- 图像

        Keyword Arguments:
            method {str} -- 卷积方法 (default: {'SAME'})
            isLogInfo {bool} -- 是否需要进度条 (default: {True})
        
        Returns:
            np.array[numFilter, imageSize] -- 返回滤波后的图像组
        """
        #- 图像数据预处理
        #@ 如果图像是单通道数据，则添加一根轴，表明单通道；多通道则不需要。适配TensorFlow卷积中图像的格式
        if type(Images)==list:
            Images = np.array(Images)

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
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if isLogInfo:
                tsg = Loggers.TFprint.TFprint("Convoluting Images...")
    
            for i in range(totalbatch):
                if isLogInfo:
                    Loggers.ProcessingBar(i + 1, totalbatch, CompleteLog='')

                Selected_Images = Images[i * batchsize : min((i + 1) * batchsize, numImages)]
                res = sess.run(conv, feed_dict={input_image:Selected_Images, input_filter:self.__Gabor_filter})
                if i == 0:
                    result = res
                else:
                    result = np.concatenate((result, res), axis=0)
            if isLogInfo:
                Loggers.TFprint.TFprint("Convoluting Images Done!", tsg)

        return result

    #- Gabor滤波器属性
    def GaborVision(self, index):
        """返回Gabor视野的延展大小，如果5*5的视野，则返回2：** * ** 左右各2
        
        Arguments:
            index {int} -- 索引
        
        Returns:
            [int] -- 视野的延展大小
        """
        return self.__Gabor_filter_size[index]

    def GaborVisionArea(self, index):
        """返回Gabor视野的面积，如果5*5的视野，则返回25
        
        Arguments:
            index {int} -- 索引
        
        Returns:
            [int] -- 视野的延展大小
        """
        return self.__Gabor_filter_area[index]

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
    
    @property
    def sumGaborVisionArea(self):
        """Gabor滤波器视野范围之和
        
        Returns:
            [int] -- Gabor滤波器视野范围之和
        """
        return sum(self.__Gabor_filter_area)

    @property
    def GaborFilterName(self, index):
        """滤波器名称

        Args:
            index (int): 索引

        Returns:
            str: 名称
        """
        return self.__Gabor_filter_name[index]

#- 附加函数
def DisplayGaborResult(Gabor_Images, figure_row=8, figure_col=16, cmap='gray'):
    """显示Gabor变换结果
    
    Arguments:
        images {np.array} -- 图像
    
    Keyword Arguments:
        figure_row {int} -- [每一行显示的图像对数] (default: {8})
        figure_col {int} -- [列数] (default: {16})
        cmap {str} -- [灰度图] (default: {'gray'})
    """
    figure_size = figure_row * figure_col
    numImages = Gabor_Images.shape[0]
    numGabor = Gabor_Images.shape[1]

    for figure_NO in range(numImages):
        #* 绘制新的 figure
        plt.figure(figure_NO)
        
        for i in range(figure_size):
            plt.subplot(figure_row, figure_col, i + 1)
            plt.imshow(Gabor_Images[figure_NO, i, :, :], cmap=cmap)
            
            #! 关闭坐标轴
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
                    
    plt.show()

def ShowDifference(kernel_size):
    """显示OpenCV库的Gabor函数和自己写的Gabor函数的区别
    
    Arguments:
        kernel_size {int} -- 核大小
    """
    import time 
    import cv2
    
    k1_time_start = time.clock()
    kernel1 = cv2.getGaborKernel((kernel_size, kernel_size), 10, 0, 10, 0.5, 0)
    k1_time_end = time.clock()
    print(kernel1.shape)
    k2_time_start = time.clock()
    kernel2 = getGaborFilter((kernel_size, kernel_size), 10, 0, 10, 0.5, 0)
    k2_time_end = time.clock()
    print("Opencv Gabor Filter Time Consumption : ", k1_time_end - k1_time_start)
    print("My Gabor Filter Time Consumption : ", k2_time_end - k2_time_start)

    x = kernel_size // 2 + 1
    y = kernel_size // 2
    print(kernel1[y][x])
    print(kernel2[y][x])
    print("Difference = ", np.argmax(kernel1 - kernel2))


#- 测试函数
def Test_MNIST(GaborClass):
    """测试MNIST数据集

    Args:
        GaborClass (class): Gabor类
    """
    #! Batchsize = 200 时，总计花费 4288.520412 s。耗时在IO，显存与内存之间数据传输
    #! Batchsize = 5000 时，总计花费 106.699461 s。节约了40倍时间。
    #! Batchsize = 10000 时，显存不足
    import Load_Datasets
    Train_X, Train_Y, Test_X, Test_Y = Load_Datasets.Preprocess_Raw_Data("./Datasets", "MNIST", True, True)
    result = GaborClass.ConvoluteImages(Test_X[0:15], batchsize=20)
    result = result.transpose(0, 3, 1, 2)
    print(result.shape)
    DisplayGaborResult(result[0:2])

def Test_Lena(GaborClass):
    """测试Lena图像

    Args:
        GaborClass (class): Gabor滤波器类
    """
    import cv2
    img = cv2.imread('./Images/Lena.jpg', 0)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    img = img[np.newaxis, :, :]
    result = GaborClass.ConvoluteImages(img)
    result = result.transpose(0, 3, 1, 2)
    DisplayGaborResult(result)
    

if __name__ == '__main__':
    ksize = (29, 29)
    Lambda = [1, 2, 3, 4]
    numTheta = 8
    Theta = [np.pi / numTheta * i for i in range(numTheta)]
    Beta = [1]
    Gamma = [0.5, 1]

    GaborClass = Gabor()
    GaborClass.setParam(ksize, Theta, Lambda, Gamma, Beta, 'b')

    Test_Lena(GaborClass)
    # Test_MNIST(GaborClass)


