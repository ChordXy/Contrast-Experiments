'''
@Author: Cabrite
@Date: 2020-03-28 16:38:00
@LastEditors: Cabrite
@LastEditTime: 2020-04-02 14:49:18
@Description: Do not edit
'''

from tensorflow.contrib.layers import xavier_initializer
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


def PrintLog(message, diff_logTime=None):
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
        Train_Feature = self.ExtractGaborFeature(self.Train_X, 5000)
        Test_Feature = self.ExtractGaborFeature(self.Test_X, 5000)
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
    def GaborTrainLabel(self):
        return self.Train_Y

    @property
    def GaborTestFeature(self):
        return self.Test_BN

    @property
    def GaborTestLabel(self):
        return self.Test_Y

    @property
    def GaborResult(self):
        return self.Train_BN, self.Train_Y, self.Test_BN, self.Test_Y

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

class DAEFeature():
    def __init__(self):
        self.getMNIST()
        self.generateDAEFeature()

    def EncoderLayer(self, Input_Layer, Input_Size, Output_Size, Activation, isTrainable=True):
        """编码层
        
        Arguments:
            Input_Layer {np.array} -- 前序层
            Input_Size {int} -- 输入大小
            Output_Size {int} -- 输出大小
            Activation {function} -- 激活函数
        
        Keyword Arguments:
            isTrainable {bool} -- [是否可训练] (default: {True})
        
        Returns:
            np.array -- 层结果
        """
        with tf.variable_scope('Encoder_Layer') as scope_encoder:
            weight = tf.get_variable('Encoder_Weight', [Input_Size, Output_Size], tf.float32, xavier_initializer(), trainable=isTrainable)
            bias = tf.get_variable('Encoder_Bias', [Output_Size], tf.float32, tf.zeros_initializer(), trainable=isTrainable)
        encoder_layer = Activation(tf.matmul(Input_Layer, weight) + bias)
        return encoder_layer
    
    def DecoderLayer(self, Input_Layer, Input_Size, Output_Size, Activation, isTrainable=True):
        """解码层
        
        Arguments:
            Input_Layer {np.array} -- 前序层
            Input_Size {int} -- 输入大小
            Output_Size {int} -- 输出大小
            Activation {function} -- 激活函数
        
        Keyword Arguments:
            isTrainable {bool} -- [是否可训练] (default: {True})
        
        Returns:
            np.array -- 层结果
        """
        with tf.variable_scope('Decoder_Layer') as scope_decoder:
            weight = tf.get_variable('Decoder_Weight', [Input_Size, Output_Size], tf.float32, xavier_initializer(), trainable=isTrainable)
            bias = tf.get_variable('Decoder_Bias', [Output_Size], tf.float32, tf.zeros_initializer(), trainable=isTrainable)
        decoder_layer = Activation(tf.matmul(Input_Layer, weight) + bias)
        return decoder_layer
    
    def generateDAEFeature(self):
        #* 高斯噪声
        gaussian = 0.02
        self.numPixels = 28 * 28
        self.numSamples = self.Train_X.shape[0]
        ############################  初始化参数  ############################
        training_epochs = 100
        batch_size = 200
        total_batch = math.ceil(self.numSamples / batch_size)
        learning_rate_dacay_init = 1e-3
        learning_rate_decay_steps = total_batch * 4
        learning_rate_decay_rates = 0.98

        ############################  初始化网络输入  ############################
        tf.reset_default_graph()
        input_Main = tf.placeholder(tf.float32, [None, self.numPixels])
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay( learning_rate=learning_rate_dacay_init, 
                                                    global_step=global_step, 
                                                    decay_steps=learning_rate_decay_steps, 
                                                    decay_rate=learning_rate_decay_rates)
 
        ############################  构建网络  ############################
        n_Hiddens = 1024
        encoder_layer = self.EncoderLayer(input_Main, self.numPixels, n_Hiddens, tf.nn.leaky_relu)
        decoder_layer = self.DecoderLayer(encoder_layer, n_Hiddens, self.numPixels, tf.nn.leaky_relu)

        #* 重建损失
        loss = tf.reduce_mean(tf.pow(tf.subtract(input_Main, decoder_layer), 2.0))

        #* 优化函数
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ############################  初始化参数  ############################
        display_step = 1
        saver = tf.train.Saver()
        model_path = './log/AE.ckpt'

        ############################  训练网络  ############################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(training_epochs):
                avg_loss = 0
                for i in range(total_batch):
                    batch_main = self.Train_X[i * batch_size : min((i + 1) * batch_size, self.numSamples), :]
                    batch_main_noise = batch_main + gaussian * np.random.randn(min(batch_size, self.numSamples - i * batch_size), self.numPixels)
                    _, ls = sess.run([optimizer, loss], feed_dict={input_Main : batch_main_noise})
                    avg_loss += ls / total_batch
                        
                if (epoch + 1) % display_step == 0:
                    message = "Epoch : " + '%04d' % (epoch + 1) + " loss = " + "{:.9f}".format(avg_loss)
                    PrintLog(message)
            
            saver.save(sess, model_path, global_step = epoch)

            print("Finished!")
            self.ae_train_feature = sess.run(encoder_layer, feed_dict={input_Main : self.Train_X})
            self.ae_test_feature = sess.run(encoder_layer, feed_dict={input_Main : self.Test_X})
            print("Features are READY!!!")

    def getMNIST(self):
        """ 获取MNIST图像
        """
        self.Train_X, self.Train_Y, self.Test_X, self.Test_Y = Load_MNIST.Preprocess_MNIST_Data("./Datasets/MNIST_Data", True, True)
        self.Train_X = np.reshape(self.Train_X, [self.Train_X.shape[0], 28 * 28])
        self.Test_X = np.reshape(self.Test_X, [self.Test_X.shape[0], 28 * 28])
        
    @property
    def DAETrainFeature(self):
        return self.ae_train_feature

    @property
    def DAETrainLabel(self):
        return self.Train_Y

    @property
    def DAETestFeature(self):
        return self.ae_test_feature

    @property
    def DAETestLabel(self):
        return self.Test_Y

    @property
    def DAEResult(self):
        return self.ae_train_feature, self.Train_Y, self.ae_test_feature, self.Test_Y


def ClassifierSVM(train_x, train_y, test_x, test_y):
    from sklearn.svm import SVC
    
    label_train = [np.argmax(sample) for sample in train_y]
    label_test = [np.argmax(sample) for sample in test_y]
    cls = SVC(kernel='rbf')
    cls.fit(train_x, label_train)
    Acc = cls.score(test_x, label_test)
    print("******************************************************")
    print("SVM Accuracy : {:5}% \r\n".format(Acc * 100))
    print("******************************************************")
    return Acc

def ClassifierMLP(train_x, train_y, test_x, test_y):
    def HiddenFullyConnectedLayer(Input_Layer, Input_Size, Output_Size, Activation, Dropout, isTrainable=True):
        """分类全链接层
        
        Arguments:
            Input_Layer {np.array} -- 前序层
            Input_Size {int} -- 输入大小
            Output_Size {int} -- 输出大小
            Activation {function} -- 激活函数
        
        Keyword Arguments:
            isTrainable {bool} -- [是否可训练] (default: {True})
        
        Returns:
            np.array -- 层结果
        """
        with tf.variable_scope('hidden_Layer', reuse=tf.AUTO_REUSE) as scope_hidden:
            weight = tf.get_variable('Hidden_Weight', [Input_Size, Output_Size], tf.float32, xavier_initializer(), trainable=isTrainable)
            bias = tf.get_variable('Hidden_Bias', [Output_Size], tf.float32, tf.zeros_initializer(), trainable=isTrainable)
        hidden_layer = Activation(tf.matmul(Input_Layer, weight) + bias)
        hidden_layer_dropout = tf.nn.dropout(hidden_layer, Dropout)
        return hidden_layer_dropout

    def SoftmaxClassifyLayer(Input_Layer, Input_Size, Output_Size, isTrainable=True):
        """Softmax分类层
        
        Arguments:
            Input_Layer {np.array} -- 前序层
            Input_Size {int} -- 输入大小
            Output_Size {int} -- 输出大小
        
        Keyword Arguments:
            isTrainable {bool} -- [是否可训练] (default: {True})
        
        Returns:
            np.array -- 层结果
        """
        with tf.variable_scope('softmax_Layer', reuse=tf.AUTO_REUSE) as scope_softmax:
            weight = tf.get_variable('Softmax_Weight', [Input_Size, Output_Size], tf.float32, xavier_initializer(), trainable=isTrainable)
            bias = tf.get_variable('Softmax_Bias', [Output_Size], tf.float32, tf.zeros_initializer(), trainable=isTrainable)
        softmax_layer = tf.nn.softmax(tf.matmul(Input_Layer, weight) + bias)
        return softmax_layer

    training_epochs = 200
    batch_size = 200
    numTrain = train_x.shape[0]
    total_batch = math.ceil(numTrain / batch_size)
    gaussian = 0.02
    learning_rate_dacay_init = 5e-5
    learning_rate_decay_steps = total_batch * 2
    learning_rate_decay_rates = 0.9

    ############################  初始化网络输入  ############################
    tf.reset_default_graph()
    n_features = train_x.shape[1]
    n_class = train_y.shape[1]
    input_Feature = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_class])
    dropout_keep_prob = tf.placeholder("float")
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay( learning_rate=learning_rate_dacay_init, 
                                                global_step=global_step, 
                                                decay_steps=learning_rate_decay_steps, 
                                                decay_rate=learning_rate_decay_rates)
 
    ############################  构建网络  ############################
    n_Hiddens = 2048
        
    #* 隐含层
    hidden_layer = HiddenFullyConnectedLayer(input_Feature, n_features, n_Hiddens, tf.nn.leaky_relu, dropout_keep_prob)
    #* 分类层
    pred = SoftmaxClassifyLayer(hidden_layer, n_Hiddens, n_class)

    #* 最终损失
    loss = tf.reduce_mean(- tf.reduce_sum(y * tf.log(pred + 1e-5), reduction_indices = 1))
 
    #* 优化函数
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #* 训练集测试函数
    correction_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

    ############################  初始化参数  ############################
    display_step = 1
        
    ############################  训练网络  ############################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            avg_loss = 0
            for i in range(total_batch):
                #* 提取每个Batch对应的数据
                batch_xs = train_x[i * batch_size : min((i + 1) * batch_size, numTrain), :]
                batch_ys = train_y[i * batch_size : min((i + 1) * batch_size, numTrain), :]
                #* 加入噪声
                batch_xs_noise = batch_xs + gaussian * np.random.randn(min(batch_size, numTrain - i * batch_size), n_features)
                #* 训练网络
                _, ls = sess.run([optimizer, loss], feed_dict={input_Feature : batch_xs_noise, y : batch_ys, dropout_keep_prob : 0.5})
                avg_loss += ls / total_batch

            if (epoch + 1) % display_step == 0:
                f_acc = accuracy.eval(feed_dict = {input_Feature : test_x, y : test_y, dropout_keep_prob : 1.})
                learn_rate = sess.run(learning_rate)
                message = "Epoch : " + '%04d' % (epoch + 1) + \
                        " Loss = " + "{:.9f}".format(avg_loss) + \
                        " Learning Rate = " + "{:.9f}".format(learn_rate) + \
                        " Final Accuracy = " + "{:.9f}".format(f_acc)

                PrintLog(message)
        print("Finished!")
        #* 测试集结果
        print("Testing Accuracy : ", accuracy.eval(feed_dict = {input_Feature : test_x, y : test_y, dropout_keep_prob : 1.}))

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

    GaborFeatures = GaborFeature(ksize, Theta, Lambda, Gamma, Beta, 'b', pool_result_size=4)
    # ClassifierSVM(*GaborFeatures.GaborResult)
    ClassifierMLP(*GaborFeatures.GaborResult)

    # DAEFeatures = DAEFeature()
    # ClassifierSVM(*DAEFeatures.DAEResult)
    # ClassifierMLP(*DAEFeatures.DAEResult)