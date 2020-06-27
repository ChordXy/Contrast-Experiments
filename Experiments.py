'''
@Author: Cabrite
@Date: 2020-03-28 16:38:00
@LastEditors: Cabrite
@LastEditTime: 2020-06-21 00:42:48
@Description: Do not edit
'''

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib import factorization
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import Load_MNIST
import Preprocess
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

#- MRAE网络
class MRAEFeature():
    def __init__(self):
        self.InitParameters()
        self.GaborMNIST(False, 5000, None, None)
        self.CaptureBlocks(False, None)
        self.Build_TiedAutoEncoderNetwork()
        self.generateMRAEFeature()

    #@ 参数初始化
    def InitParameters(self):
        #- 读取MNIST图像
        #* 输入图像大小
        self.ImageSize = (28, 28)
        #* 类别数
        self.n_class = 10
        #* 读入图像
        self.getMNIST()

        #- Gabor 参数
        ksize = (29, 29)
        Lambda = [1, 2, 3, 4]
        numTheta = 8
        Theta = [np.pi / numTheta * i for i in range(numTheta)]
        Beta = [1]
        Gamma = [0.5, 1]
        RI_Part = 'b'

        #- 获取Gabor滤波器组
        self.Gabor_Filter = Preprocess.Gabor()
        self.Gabor_Filter.setParam(ksize, Theta, Lambda, Gamma, Beta, RI_Part)
        #* Gabor滤波器视场总和（用于Siamese输入）
        self.sumGaborVisionArea = self.Gabor_Filter.sumGaborVisionArea

        #- 图像块定义及参数
        #* 图像块大小
        self.ImageBlockSize = (11, 11)
        #* 采样数
        self.numSamples = 400000
        #* 图像块像素总数
        self.numPixels = self.ImageBlockSize[0] * self.ImageBlockSize[1]
        #* PCA白化
        self.Whiten = True
        
        #- 其他参数
        #* 打印Log时的步长
        self.log_display_num = self.DisplayStepCount()
        #* ？？？
        self.numBlocksOfImage = (self.ImageSize[0] - self.ImageBlockSize[0] + 1) * (self.ImageSize[1] - self.ImageBlockSize[1] + 1)


    #@ 数据预处理
    def getMNIST(self):
        """ 获取MNIST图像
        """
        self.Train_X, self.Train_Y, self.Test_X, self.Test_Y = Preprocess.Preprocess_MNIST_Data("./Datasets/MNIST_Data", True, True)

    def GaborMNIST(self, isLoadFile, batchsize=1000, savefile_Train_Gabor='Gabored_MNIST_Images_Train.npy', savefile_Test_Gabor='Gabored_MNIST_Images_Test.npy'):
        """Gabor MNIST 图像集，全图Gabor
        
        Arguments:
            isLoadFile {bool} -- 是否读取已存在的文件
        
        Keyword Arguments:
            savefile_Train_Gabor {str} -- 训练集Gabor路径 (default: {'Gabored_MNIST_Images_Train.npy'})
            savefile_Test_Gabor {str} -- 测试集Gabor路径 (default: {'Gabored_MNIST_Images_Test.npy'})
        
        Returns:
            np.array, np.array -- Gabor结果
        """
        if isLoadFile==False:
            self.Train_X_Gabor = Preprocess.GaborAllImages(self.Gabor_Filter, self.Train_X, batchsize=batchsize, isSavingData=savefile_Train_Gabor)
            self.Test_X_Gabor = Preprocess.GaborAllImages(self.Gabor_Filter, self.Test_X, batchsize=batchsize, isSavingData=savefile_Test_Gabor)
        else:
            self.Train_X_Gabor = Preprocess.LoadGaborImages(savefile_Train_Gabor)
            self.Test_X_Gabor = Preprocess.LoadGaborImages(savefile_Test_Gabor)

    def CaptureBlocks(self, isLoadFile, savefile_ImageBlocks=['ImageBlocks.npy', 'ImageBlocksGabor.npy']):
        """采样
        
        Arguments:
            isLoadFile {bool} -- 是否读取已存在文件
        
        Keyword Arguments:
            savefile_ImageBlocks {list} -- 保存路径 (default: {['ImageBlocks.npy', 'ImageBlocksGabor.npy']})
        """
        if isLoadFile==False:
            self.Image_Blocks, self.Image_Blocks_Gabor = Preprocess.RandomSamplingImageBlocks(self.Train_X, self.Train_X_Gabor, self.Gabor_Filter, self.ImageBlockSize, self.numSamples, isSavingData=savefile_ImageBlocks, isLog=self.log_display_num)
        else:
            self.Image_Blocks, self.Image_Blocks_Gabor = Preprocess.LoadRandomImageBlocks(savefile_ImageBlocks)

        self.Image_Blocks, self.Whiten_Average, self.Whiten_U = Preprocess.PCA_Whiten(self.Image_Blocks, self.Whiten)
        

    #@ AE网络
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
    
    def TiedEncoderDecoderLayer(self, Input_Layer, Input_Size, Hidden_Size, Activation_Encoder, isTrainable=True):
        """绑定编、解码层
        
        Arguments:
            Input_Layer {np.array} -- 前序层
            Input_Size {int} -- 输入大小
            Hidden_Size {int} -- 隐含层大小
            Activation_Encoder {function} -- 编码器激活函数
            Activation_Decoder {function} -- 解码器激活函数
        
        Keyword Arguments:
            isTrainable {bool} -- [是否可训练] (default: {True})
        
        Returns:
            tf.tensor, tf.tensor -- 编码层结果，解码层结果
        """
        with tf.variable_scope('TiedEncoderDecoder_Layer') as scope:
            weight = tf.get_variable('Tied_Weight', [Input_Size, Hidden_Size], tf.float32, xavier_initializer(), trainable=isTrainable)
            bias_en = tf.get_variable('Tied_Encoder_Bias', [Hidden_Size], tf.float32, tf.zeros_initializer(), trainable=isTrainable)
            bias_de = tf.get_variable('Tied_Decoder_Bias', [Input_Size], tf.float32, tf.zeros_initializer(), trainable=isTrainable)
        encoder_layer = Activation_Encoder(tf.matmul(Input_Layer, weight) + bias_en)
        decoder_layer = tf.matmul(encoder_layer, tf.transpose(weight)) + bias_de
        return encoder_layer, decoder_layer

    def SiameseLayer(self, Input_Layer, Input_Size, Output_Size, Activation, isTrainable=True):
        """Siamese层
        
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
        with tf.variable_scope('Siamese_Layer') as scope_siamese:
            weight = tf.get_variable('Siamese_Weight', [Input_Size, Output_Size], tf.float32, xavier_initializer(), trainable=isTrainable)
            bias = tf.get_variable('Siamese_Bias', [Output_Size], tf.float32, tf.zeros_initializer(), trainable=isTrainable)
        siamese_layer = Activation(tf.matmul(Input_Layer, weight) + bias)
        return siamese_layer

    def Build_TiedAutoEncoderNetwork(self):
        """带Siamese旁支、绑定权重的AE网络
        """
        #* 重建比重
        reconstruction_reg = 0.5
        #* 相似度量比重
        measurement_reg = 0.1
        #* 稀疏性比重
        sparse_reg = 0.1
        #* 高斯噪声
        gaussian = 0.02

        ############################  初始化参数  ############################
        training_epochs = 100
        batch_size = 200
        total_batch = math.ceil(self.numSamples / batch_size)
        learning_rate_dacay_init = 1e-2
        learning_rate_decay_steps = total_batch * 4
        learning_rate_decay_rates = 0.95

        ############################  初始化网络输入  ############################
        tf.reset_default_graph()
        input_Main = tf.placeholder(tf.float32, [None, self.numPixels])
        input_Siamese = tf.placeholder(tf.float32, [None, self.sumGaborVisionArea])
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay( learning_rate=learning_rate_dacay_init, 
                                                    global_step=global_step, 
                                                    decay_steps=learning_rate_decay_steps, 
                                                    decay_rate=learning_rate_decay_rates)
 
        ############################  构建网络  ############################
        n_Hiddens = 1024
        encoder_layer, decoder_layer = self.TiedEncoderDecoderLayer(input_Main, self.numPixels, n_Hiddens, tf.nn.leaky_relu)
        siamese_layer = self.SiameseLayer(input_Siamese, self.sumGaborVisionArea, n_Hiddens, tf.nn.leaky_relu)

        #* 重建损失
        loss_reconstruction = tf.reduce_mean(tf.pow(tf.subtract(input_Main, decoder_layer), 2.0))
        #* 度量损失
        loss_measurement = tf.reduce_mean(tf.abs(tf.subtract(encoder_layer, siamese_layer)))
        #* 稀疏性
        loss_sparse = tf.reduce_mean(tf.abs(siamese_layer))
        #* 最终损失
        loss = reconstruction_reg * loss_reconstruction + measurement_reg * loss_measurement + sparse_reg * loss_sparse
        #* 优化函数
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ############################  初始化参数  ############################
        display_step = 1
        saver = tf.train.Saver()
        model_path = './log/mRAE.ckpt'

        ############################  训练网络  ############################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(training_epochs):
                avg_loss = 0
                for i in range(total_batch):
                    #* 提取每个Batch对应的数据
                    batch_main = self.Image_Blocks[i * batch_size : min((i + 1) * batch_size, self.numSamples), :]
                    batch_siamese = self.Image_Blocks_Gabor[i * batch_size : min((i + 1) * batch_size, self.numSamples), :]
                    #* 加入噪声
                    batch_main_noise = batch_main + gaussian * np.random.randn(min(batch_size, self.numSamples - i * batch_size), self.numPixels)
                    #* 训练网络
                    _, ls = sess.run([optimizer, loss], feed_dict={input_Main : batch_main_noise, input_Siamese : batch_siamese})
                    avg_loss += ls / total_batch
                    
                if (epoch + 1) % display_step == 0:
                    message = "Epoch : " + '%04d' % (epoch + 1) + " loss = " + "{:.9f}".format(avg_loss)
                    Preprocess.PrintLog(message)
                    save_path = saver.save(sess, model_path, global_step = epoch)

            print("Finished!")
            save_path = saver.save(sess, model_path)
            print("Model saved in file : ", save_path)

    def Build_AutoEncoderNetwork(self):
        """带Siamese旁支的AE网络
        """
        #* 重建比重
        reconstruction_reg = 0.5
        #* 相似度量比重
        measurement_reg = 0.1
        #* 稀疏性比重
        sparse_reg = 0.1
        #* 高斯噪声
        gaussian = 0.02

        ############################  初始化参数  ############################
        training_epochs = 100
        batch_size = 200
        total_batch = math.ceil(self.numSamples / batch_size)
        learning_rate_dacay_init = 0.1
        learning_rate_decay_steps = total_batch * 4
        learning_rate_decay_rates = 0.98

        ############################  初始化网络输入  ############################
        tf.reset_default_graph()
        input_Main = tf.placeholder(tf.float32, [None, self.numPixels])
        input_Siamese = tf.placeholder(tf.float32, [None, self.sumGaborVisionArea])
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay( learning_rate=learning_rate_dacay_init, 
                                                    global_step=global_step, 
                                                    decay_steps=learning_rate_decay_steps, 
                                                    decay_rate=learning_rate_decay_rates)
 
        ############################  构建网络  ############################
        n_Hiddens = 1024
        encoder_layer = self.EncoderLayer(input_Main, self.numPixels, n_Hiddens, tf.nn.leaky_relu)
        decoder_layer = self.DecoderLayer(encoder_layer, n_Hiddens, self.numPixels, tf.nn.leaky_relu)
        siamese_layer = self.SiameseLayer(input_Siamese, self.sumGaborVisionArea, n_Hiddens, tf.nn.leaky_relu)

        #* 重建损失
        loss_reconstruction = tf.reduce_mean(tf.pow(tf.subtract(input_Main, decoder_layer), 2.0))
        #* 度量损失
        loss_measurement = tf.reduce_mean(tf.abs(tf.subtract(encoder_layer, siamese_layer)))
        #* 稀疏性
        loss_sparse = tf.reduce_mean(tf.abs(siamese_layer))
        #* 最终损失
        loss = reconstruction_reg * loss_reconstruction + measurement_reg * loss_measurement + sparse_reg * loss_sparse
        #* 优化函数
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ############################  初始化参数  ############################
        display_step = 1
        saver = tf.train.Saver()
        model_path = './log/mRAE.ckpt'

        ############################  训练网络  ############################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(training_epochs):
                avg_loss = 0
                for i in range(total_batch):
                    #* 提取每个Batch对应的数据
                    batch_main = self.Image_Blocks[i * batch_size : min((i + 1) * batch_size, self.numSamples), :]
                    batch_siamese = self.Image_Blocks_Gabor[i * batch_size : min((i + 1) * batch_size, self.numSamples), :]
                    #* 加入噪声
                    batch_main_noise = batch_main + gaussian * np.random.randn(min(batch_size, self.numSamples - i * batch_size), self.numPixels)
                    #* 训练网络
                    _, ls = sess.run([optimizer, loss], feed_dict={input_Main : batch_main_noise, input_Siamese : batch_siamese})
                    avg_loss += ls / total_batch
                    
                if (epoch + 1) % display_step == 0:
                    message = "Epoch : " + '%04d' % (epoch + 1) + " loss = " + "{:.9f}".format(avg_loss)
                    Preprocess.PrintLog(message)
                    save_path = saver.save(sess, model_path, global_step = epoch)

            print("Finished!")
            save_path = saver.save(sess, model_path)
            print("Model saved in file : ", save_path)

    def Display_TiedReconstruction(self, numImages):
        """显示重建结果
        
        Arguments:
            numImages {int} -- 重建的图像数量
        """
        tf.reset_default_graph()

        ############################  构建网络  ############################
        n_Hiddens = 1024
        input_Main = tf.placeholder(tf.float32, [None, self.numPixels])
        encoder_layer, decoder_layer = self.TiedEncoderDecoderLayer(input_Main, self.numPixels, n_Hiddens, tf.nn.leaky_relu)

        saver = tf.train.Saver()
        model_path = './log/mRAE.ckpt'

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)

            batch_xs = self.Image_Blocks[0 : numImages]
            output_val = sess.run([decoder_layer], feed_dict = {input_Main : batch_xs})
            
            batch_xs = np.reshape(batch_xs, [numImages, *self.ImageBlockSize])
            output_val = np.reshape(output_val, [numImages, *self.ImageBlockSize])

            self.DisplayReconstructionResult(batch_xs, output_val)

    def Display_Reconstruction(self, numImages):
        """显示重建结果
        
        Arguments:
            numImages {int} -- 重建的图像数量
        """
        tf.reset_default_graph()

        ############################  构建网络  ############################
        n_Hiddens = 1024
        input_Main = tf.placeholder(tf.float32, [None, self.numPixels])
        encoder_layer = self.EncoderLayer(input_Main, self.numPixels, n_Hiddens, tf.nn.leaky_relu)
        decoder_layer = self.DecoderLayer(encoder_layer, n_Hiddens, self.numPixels, tf.nn.leaky_relu)

        saver = tf.train.Saver()
        model_path = './log/mRAE.ckpt'

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)

            batch_xs = self.Image_Blocks[0 : numImages]
            output_val = sess.run([decoder_layer], feed_dict = {input_Main : batch_xs})
            
            batch_xs = np.reshape(batch_xs, [numImages, *self.ImageBlockSize])
            output_val = np.reshape(output_val, [numImages, *self.ImageBlockSize])

            self.DisplayReconstructionResult(batch_xs, output_val)


    #@ MLP 分类网络
    def HiddenFullyConnectedLayer(self, Input_Layer, Input_Size, Output_Size, Activation, Dropout, isTrainable=True):
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

    def SoftmaxClassifyLayer(self, Input_Layer, Input_Size, Output_Size, isTrainable=True):
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

    def MLPLayer(self, Input_Layer, Input_Size, Output_Size, isTrainable=True):
        """MLP分类层
        
        Arguments:
            Input_Layer {np.array} -- 前序层
            Input_Size {int} -- 输入大小
            Output_Size {int} -- 输出大小
        
        Keyword Arguments:
            isTrainable {bool} -- [是否可训练] (default: {True})
        
        Returns:
            np.array -- 层结果
        """
        with tf.variable_scope('mlp_Layer', reuse=tf.AUTO_REUSE) as scope_mlp:
            weight = tf.get_variable('MLP_Weight', [Input_Size, Output_Size], tf.float32, xavier_initializer(), trainable=isTrainable)
            bias = tf.get_variable('MLP_Bias', [Output_Size], tf.float32, tf.zeros_initializer(), trainable=isTrainable)
        mlp_layer = tf.matmul(Input_Layer, weight) + bias
        return mlp_layer

    def ExtractEncoderFeature(self, Data, Data_Gabor):
        #- 初始化参数
        block_row, block_col = self.ImageBlockSize
        numData, Data_row, Data_col = Data.shape
        numRow = Data_row - block_row + 1
        numCol = Data_col - block_col + 1
        ksize = [1, numRow // 2, numCol // 2, 1]
        stride = [1, numRow - numRow // 2, numCol - numCol //2, 1]

        batchsize = 1000
        totalbatch = math.ceil(numData / batchsize)
        display_step = 1

        #! 演示分割结果
        # display_no = 0
        # self.DisplaySplitResult(Data[display_no], np.reshape(Splited_Data[display_no, :, :], [numRow * numCol, block_row, block_col]), numRow, numCol)
        
        #- 提取高维特征并池化，重新组合
        tf.reset_default_graph()
        n_Hiddens = 1024
        # Encodered_Data = np.zeros([numData, 4 * n_Hiddens * 2])
        Encodered_Data = np.zeros([numData, 4 * n_Hiddens])

        #* 输入
        input_Main = tf.placeholder(tf.float32, [batchsize, numRow * numCol, self.numPixels])
        # input_Siamese = tf.placeholder(tf.float32, [batchsize, numRow * numCol, self.sumGaborVisionArea])
        #* 改变形状，适应批量乘法
        input_Main_Re = tf.reshape(input_Main, [batchsize * numRow * numCol, self.numPixels])
        # input_Siamese_Re = tf.reshape(input_Siamese, [batchsize * numRow * numCol, self.sumGaborVisionArea])
        #* 特征编码
        # encoder_layer = self.EncoderLayer(input_Main_Re, self.numPixels, n_Hiddens, tf.nn.leaky_relu, False)
        encoder_layer, _ = self.TiedEncoderDecoderLayer(input_Main_Re, self.numPixels, n_Hiddens, tf.nn.leaky_relu, False)
        # siamese_layer = self.SiameseLayer(input_Siamese_Re, self.sumGaborVisionArea, n_Hiddens, tf.nn.leaky_relu, False)
        #* 变换形状
        # concat_result = tf.concat([encoder_layer, siamese_layer], 1)
        # concat_result_Re = tf.reshape(concat_result, [batchsize, numRow, numCol, 2 * n_Hiddens])
        concat_result_Re = tf.reshape(encoder_layer, [batchsize, numRow, numCol, n_Hiddens])
        #* 均值池化
        maxpool = tf.nn.avg_pool(concat_result_Re, ksize, stride, 'VALID')
        #* 变换形状，首尾拼接
        # reshaped_maxpool = tf.reshape(maxpool, [batchsize, 4 * 2 * n_Hiddens])
        reshaped_maxpool = tf.reshape(maxpool, [batchsize, 4 * n_Hiddens])

        saver = tf.train.Saver()
        model_path = './log/mRAE.ckpt'

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            tsg = self.PrintLog("Extracting High Dimensional Features...")
            for i in range(totalbatch):
                Splited_Image, Splited_Image_Gabor = Preprocess.BatchFullySamplingImages(Data[i * batchsize : min((i + 1) * batchsize, numData)],
                                                                        Data_Gabor[i * batchsize : min((i + 1) * batchsize, numData)],
                                                                        self.Gabor_Filter,
                                                                        self.ImageBlockSize,
                                                                        Whiten=True,
                                                                        Whiten_Average=self.Whiten_Average,
                                                                        Whiten_U=self.Whiten_U)

                Encodered_Data[i * batchsize : min((i + 1) * batchsize, numData), :] = sess.run(reshaped_maxpool, feed_dict = {input_Main : Splited_Image})
                # Encodered_Data[i * batchsize : min((i + 1) * batchsize, numData), :] = sess.run(reshaped_maxpool, feed_dict = {input_Main : Splited_Image, input_Siamese : Splited_Image_Gabor})
                if (i + 1) % display_step == 0:
                    message = "Extracting High Dimensional Features : {}/{}".format(min((i + 1) * batchsize, numData), numData)
                    self.PrintLog(message)
            self.PrintLog("Extracting High Dimensional Features Done!", tsg)

        return Encodered_Data

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
        
    def generateMRAEFeature(self):
        """建立分类网络
        """
        training_epochs = 500
        batch_size = 200
        numTrain = self.Train_X.shape[0]
        total_batch = math.ceil(numTrain / batch_size)
        gaussian = 0.02
        learning_rate_dacay_init = 1e-4
        learning_rate_decay_steps = total_batch * 2
        learning_rate_decay_rates = 0.95

        Train_X_feature = self.ExtractEncoderFeature(self.Train_X, self.Train_X_Gabor)
        del self.Train_X_Gabor
        gc.collect()

        Test_X_feature = self.ExtractEncoderFeature(self.Test_X, self.Test_X_Gabor)
        del self.Test_X_Gabor
        gc.collect()

        self.Train_X_feature, self.Test_X_feature = self.FeatureReduction(Train_X_feature, Test_X_feature)

    @property
    def MRAETrainFeature(self):
        return self.Train_X_feature

    @property
    def MRAETrainLabel(self):
        return self.Train_Y

    @property
    def MRAETestFeature(self):
        return self.Test_X_feature

    @property
    def MRAETestLabel(self):
        return self.Test_Y

    @property
    def MRAEResult(self):
        return self.Train_X_feature, self.Train_Y, self.Test_X_feature, self.Test_Y

    #@ 附加函数
    def DisplayStepCount(self):
        """Log显示的时候步长step
        
        Returns:
            int -- 打Log时，产生的步长
        """
        i = 0
        temp = self.numSamples
        while temp > 0:
            i += 1
            temp = temp // 10
        return max(1, pow(10, i - 2))

    def DisplayReconstructionResult(self, origin, reconstruction, figure_row=4, figure_col=8):
        """显示重建前后的图像。 奇数行是原图，偶数行是重建图
        
        Arguments:
            origin {np.array[num, row, col]} -- 源图像，需要进行形状变换
            reconstruction {np.array[num, row, col]} -- `重建图像，需要进行形状变换`
        
        Keyword Arguments:
            figure_row {int} -- [每一行显示的图像对数] (default: {4})
            figure_col {int} -- [列数] (default: {8})
        """
        
        figure_size = figure_row * figure_col
        numImages = origin.shape[0]
        numFigure = int(numImages / figure_size) + 1
        image_origin_count = 0
        image_reconstruction_count = 0
        Done_flag = False

        for figure_NO in range(numFigure):
            #! 防止出现空白的 figure
            if Done_flag == True or image_reconstruction_count == numImages:
                break
            #* 绘制新的 figure
            plt.figure(figure_NO)
            #* i需要循环 原图，重建图   因此需要乘2
            for i in range(figure_row * 2):
                if Done_flag == True:
                    break
                for j in range(figure_col):
                    if image_reconstruction_count == numImages:
                        Done_flag = True
                        break
                    
                    if i % 2 == 0 and image_origin_count < numImages:
                        plt.subplot(figure_row * 2, figure_col, i * figure_col + j + 1)
                        #! 关闭坐标轴
                        plt.imshow(origin[image_origin_count])
                        plt.xticks([])
                        plt.yticks([])
                        plt.axis('off')
                        image_origin_count += 1
                    if i % 2 == 1:
                        plt.subplot(figure_row * 2, figure_col, i * figure_col + j + 1)
                        plt.imshow(reconstruction[image_reconstruction_count])
                        #! 关闭坐标轴
                        plt.xticks([])
                        plt.yticks([])
                        plt.axis('off')
                        image_reconstruction_count += 1

        plt.show()

    def DisplaySplitResult(self, origin_image, split_image, rows, cols):
        """显示图像分割结果
        
        Arguments:
            origin_image {np.array} -- 源图像
            split_image {np.array} -- 分割结果
            rows {int} -- 分割后的行数
            cols {int} -- 分割后的列数
        """
        plt.figure(0)
        plt.imshow(origin_image)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        plt.figure(1)
        for i in range(rows):
            for j in range(cols):
                plt.subplot(rows, cols, i * cols + j + 1)
                plt.imshow(split_image[i * cols + j])
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
        plt.show()

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
        flattened_maxpool = tf.layers.flatten(maxpool)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tsg = self.PrintLog("Extracting Features...")
    
            for i in range(totalbatch):
                self.PrintLog("Extracting Features... {}/{}".format(min(i * batchsize, numImages), numImages))

                Selected_Images = Images[i * batchsize : (i + 1) * batchsize]
                result_pool = sess.run(flattened_maxpool, feed_dict={input_image:Selected_Images, input_filter:self.__Gabor_filter})
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

#- DAE网络
class DAEFeature():
    def __init__(self, hiddens = 1024):
        self.nhiddens = hiddens
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
        n_Hiddens = self.nhiddens
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

#- DAE网络 - Block
class BlockDAEFeature():
    def __init__(self, hiddens = 1024):
        self.nhiddens = hiddens
        self.getMNIST()
        self.RandomSamplingImageBlocks((11, 11), 400000)
        self.TrainDAEFeature()
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

    def TrainDAEFeature(self):
        #* 高斯噪声
        gaussian = 0.02
        self.numPixels = 11 * 11
        self.numSamples = self.Train_X_Block.shape[0]
        ############################  初始化参数  ############################
        training_epochs = 100
        batch_size = 1000
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
        n_Hiddens = self.nhiddens
        encoder_layer = self.EncoderLayer(input_Main, self.numPixels, n_Hiddens, tf.nn.leaky_relu)
        decoder_layer = self.DecoderLayer(encoder_layer, n_Hiddens, self.numPixels, tf.nn.leaky_relu)

        #* 重建损失
        loss = tf.reduce_mean(tf.pow(tf.subtract(input_Main, decoder_layer), 2.0))

        #* 优化函数
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ############################  初始化参数  ############################
        display_step = 5
        saver = tf.train.Saver()
        model_path = './slog/DAE.ckpt'

        ############################  训练网络  ############################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(training_epochs):
                avg_loss = 0
                for i in range(total_batch):
                    batch_main = self.Train_X_Block[i * batch_size : min((i + 1) * batch_size, self.numSamples), :]
                    batch_main_noise = batch_main + gaussian * np.random.randn(min(batch_size, self.numSamples - i * batch_size), self.numPixels)
                    _, ls = sess.run([optimizer, loss], feed_dict={input_Main : batch_main_noise})
                    avg_loss += ls / total_batch
                        
                if (epoch + 1) % display_step == 0:
                    message = "Epoch : " + '%04d' % (epoch + 1) + " loss = " + "{:.9f}".format(avg_loss)
                    PrintLog(message)
            print("Finished!")
            save_path = saver.save(sess, model_path)

    def generateDAEFeature(self):
        self.Train_X_Encodered = self.ExtractDAEFeature(self.Train_X, (11, 11))
        self.Test_X_Encodered = self.ExtractDAEFeature(self.Test_X, (11, 11))

    def ExtractDAEFeature(self, Data, BlockSize):
        block_row, block_col = BlockSize
        numData, Data_row, Data_col = Data.shape
        numRow = Data_row - block_row + 1
        numCol = Data_col - block_col + 1
        ksize = [1, numRow // 2, numCol // 2, 1]
        stride = [1, numRow - numRow // 2, numCol - numCol //2, 1]
        self.numPixels = 11 * 11

        batchsize = 200
        totalbatch = math.ceil(numData / batchsize)
        display_step = 1000

        #- 提取高维特征并池化，重新组合
        tf.reset_default_graph()
        n_Hiddens = self.nhiddens
        Encodered_Data = np.zeros([numData, 4 * n_Hiddens])

        #* 输入
        input_Main = tf.placeholder(tf.float32, [batchsize, numRow * numCol, self.numPixels])
        #* 改变形状，适应批量乘法
        input_Main_Re = tf.reshape(input_Main, [batchsize * numRow * numCol, self.numPixels])
        #* 特征编码
        encoder_layer = self.EncoderLayer(input_Main_Re, self.numPixels, n_Hiddens, tf.nn.leaky_relu, False)
        #* 变换形状
        concat_result_Re = tf.reshape(encoder_layer, [batchsize, numRow, numCol, n_Hiddens])
        #* 均值池化
        avgpool = tf.nn.avg_pool(concat_result_Re, ksize, stride, 'VALID')
        #* 变换形状，首尾拼接
        reshaped_avgpool = tf.reshape(avgpool, [batchsize, 4 * n_Hiddens])

        saver = tf.train.Saver()
        model_path = './slog/DAE.ckpt'

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            tsg = PrintLog("Extracting High Dimensional Features...")
            for i in range(totalbatch):
                Splited_Image = self.BatchFullySamplingImages(Data[i * batchsize : min((i + 1) * batchsize, numData)], BlockSize)
                Encodered_Data[i * batchsize : min((i + 1) * batchsize, numData), :] = sess.run(reshaped_avgpool, feed_dict = {input_Main : Splited_Image})
                
                if (i + 1) % display_step == 0:
                    message = "Extracting High Dimensional Features : {}/{}".format(min((i + 1) * batchsize, numData), numData)
                    PrintLog(message)
            PrintLog("Extracting High Dimensional Features Done!", tsg)
        return Encodered_Data

    def getMNIST(self):
        """ 获取MNIST图像
        """
        self.Train_X, self.Train_Y, self.Test_X, self.Test_Y = Load_MNIST.Preprocess_MNIST_Data("./Datasets/MNIST_Data", True, True)

    def RandomSamplingImageBlocks(self, ImageBlockSize, numSample, isLog=5000):
        tsg = PrintLog("Image Blocks Capturing...")

        #* 参数初始化
        numImages, image_rows, image_cols = self.Train_X.shape
        block_rows, block_cols = ImageBlockSize
        half_block_rows, half_block_cols = block_rows // 2, block_cols // 2
        #* 预定义返回数组
        Image_Block = np.zeros([numSample, block_rows, block_cols])

        #* 随机生成选取的图片序号
        num_CaptureImage = np.random.randint(numImages, size=[numSample])
        for index, NumberOfBlocks in enumerate(num_CaptureImage):
            #- 提取原图像
            #* 获取第n个图片及其Gabor图像
            Selected_Image = self.Train_X[NumberOfBlocks, :, :]
            #* 提取合法区域内的图像
            Image_In_Range = Selected_Image[half_block_rows : image_rows - half_block_rows, half_block_cols : image_cols - half_block_cols]
            #* 获取合法区域内非零点坐标
            nonZero_Position = Image_In_Range.nonzero()
            #* 在合法坐标内随机选择一对坐标
            Selected_Coordinate = np.random.randint(len(nonZero_Position[0]))
            #* 获取原图上的坐标
            Selected_x = nonZero_Position[0][Selected_Coordinate] + half_block_cols
            Selected_y = nonZero_Position[1][Selected_Coordinate] + half_block_rows
            #* 截取图像，存入数组
            Image_Block[index] = Selected_Image[Selected_y - half_block_rows : Selected_y + half_block_rows + 1, Selected_x - half_block_cols: Selected_x + half_block_cols + 1]

            #- 显示日志信息
            if isLog > 0 and (index + 1) % isLog == 0:
                message = "Now Capturing Image Blocks and Gabor Images... " + str(index + 1) + '/'+ str(numSample)
                PrintLog(message)

        #- 调整形状，将Image_Block 从 [numSample, block_rows, block_cols] 转成 [numSample, block_rows * block_cols]
        self.Train_X_Block = np.reshape(Image_Block, [numSample, block_rows * block_cols])

        PrintLog("Image Blocks and Gabor Images Capturing Done!", tsg)

    def BatchFullySamplingImages(self, Images, ImageBlockSize):
        block_row, block_col = ImageBlockSize
        numImages, Image_row, Image_col = Images.shape

        numRow = Image_row - block_row + 1
        numCol = Image_col - block_col + 1

        Splited_Image = np.zeros([numImages, numRow * numCol, block_row * block_col])

        for i in range(numRow):
            for j in range(numCol):
                #- 顺序截取图像块
                img = Images[:, i : i + block_row, j : j + block_col]
                Splited_Image[:, i * numCol + j, :] = np.reshape(img, [numImages, block_row * block_col])

        return Splited_Image


    @property
    def DAETrainFeature(self):
        return self.Train_X_Encodered

    @property
    def DAETrainLabel(self):
        return self.Train_Y

    @property
    def DAETestFeature(self):
        return self.Test_X_Encodered

    @property
    def DAETestLabel(self):
        return self.Test_Y

    @property
    def DAEResult(self):
        return self.Train_X_Encodered, self.Train_Y, self.Test_X_Encodered, self.Test_Y


#- 监督学习：SVM分类器
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

#- 监督学习：MLP分类器
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
        acc = accuracy.eval(feed_dict = {input_Feature : test_x, y : test_y, dropout_keep_prob : 1.})
        print("Testing Accuracy : ", acc)
        return acc

#- 降维
def DimensionReduction(data, Targeted_Dimension = 2, method = 0):
    if data.shape[1] <= Targeted_Dimension:
        return data
    if method == 0:
        Reduced_Data = PCA(n_components=Targeted_Dimension).fit_transform(data)
    else:
        Reduced_Data = TSNE(n_components=Targeted_Dimension).fit_transform(data)
    return Reduced_Data

#- 无监督学习：k均值CPU
def ClassifierKMeans(data, n_class = 10):
    estimator = KMeans(n_clusters=n_class)
    estimator.fit(data)

#- 无监督学习：k均值GPU
def ClassifierKMeansKNN(data, Cluster_Centers = 25, Targeted_Dimension = 2, method = 0):
    ################################### 数据降维 ###################################
    if Targeted_Dimension > 0:
        Train_X = DimensionReduction(data[0], Targeted_Dimension, method)
        Train_Y = data[1]
        Test_X = DimensionReduction(data[2], Targeted_Dimension, method)
        Test_Y = data[3]
    else:
        Train_X, Train_Y, Test_X, Test_Y = data
    ################################### 参数初始化 ###################################
    # 训练次数
    epochs = 300
    # 每一批的样本数
    batch_size = 1024
    # 特征数
    num_features = Train_X.shape[1]
    # 显示步长
    display_epoch = 500
    # 类别数
    n_class = 10
    acc = 0

    ################################### 网络参数 ###################################
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.float32, shape=[None, n_class])

    # 创建 K-Means模型
    _, cluster_idx, scores, _, init_op, train_op = factorization.KMeans(inputs=X, num_clusters=Cluster_Centers, distance_metric='cosine', use_mini_batch=True).training_graph()
    cluster_idx = cluster_idx[0]
    avg_distance = tf.reduce_mean(scores)

    ################################### 训练网络 ###################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_op, feed_dict={X: Train_X})

        #* 聚类
        for i in range(epochs):
            _, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X: Train_X})
            if (i + 1) % display_epoch == 0:
                PrintLog("Step %i, Avg Distance: %f" % (i + 1, d))

        #* KNN投票
        counts = np.zeros(shape=(Cluster_Centers, n_class))
        for i in range(len(idx)):
            counts[idx[i]] += Train_Y[i]
        labels_map = tf.convert_to_tensor([np.argmax(c) for c in counts])

        #* 建立评测网络
        cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
        correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = sess.run(accuracy_op, feed_dict={X: Test_X, Y: Test_Y})
        print("Test Accuracy:", acc)
    return acc


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

#- 文件函数
def PrintToFile(results, filename):
    results = np.transpose(results)
    with open(filename, "w+") as savefile:
        for line in results:
            for res in line:
                savefile.write("{:.4}\t".format(res))
            savefile.write('\r\n')
            

if __name__ == "__main__":
    ksize = (11, 11)
    Lambda = [1, 2, 3, 4]
    numTheta = 8
    Theta = [np.pi / numTheta * i for i in range(numTheta)]
    Beta = [1]
    Gamma = [0.5, 1]
    
    
    #- Gabor vs mrDAE - Dimension
    results = []
    prs = [1]
    ds = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    for ps in prs:
        res = []
        GaborFeatures = GaborFeature(ksize, Theta, Lambda, Gamma, Beta, 'b', pool_result_size=ps)
    #     #* 监督学习
        res.append(ClassifierSVM(*GaborFeatures.GaborResult))
        res.append(ClassifierMLP(*GaborFeatures.GaborResult))
    #     #* 无监督学习
        for d in ds:
            print("************************************************")
            print("  Gabor(s = {}) vs mrDAE -- k = [25], d = {} ".format(ps, d))
            print("************************************************")
            res.append(ClassifierKMeansKNN(GaborFeatures.GaborResult, 25, d))
        results.append(res)
    PrintToFile(results, "Change of d - Gabor.txt")

    #- Gabor vs mrDAE - Cluster Centers
    #* Gabor
    ds = [16, 32, 64, 128]
    prs = [1]
    ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for d in ds:
        results = []
        for ps in prs:
            res = []
            GaborFeatures = GaborFeature(ksize, Theta, Lambda, Gamma, Beta, 'b', pool_result_size=ps)
            #* 无监督学习
            for k in ks:
                print("************************************************")
                print("  Gabor(s = {}) vs mrDAE -- k = {}, d = [{}] ".format(ps, k, d))
                print("************************************************")
                res.append(ClassifierKMeansKNN(GaborFeatures.GaborResult, k, d))
            results.append(res)
        PrintToFile(results, "Change of k (d={}) - Gabor.txt".format(d))

    #* mrDAE
    # ds = [16, 32, 64, 128]
    # ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # mrDAEFeatures = MRAEFeature()
    # results = []
    # for d in ds:
    #     res = []
    #     for k in ks:
    #         print("************************************************")
    #         print("  Gabor vs mrDAE -- k = {}, d = [{}] ".format(k, d))
    #         print("************************************************")
    #         res.append(ClassifierKMeansKNN(mrDAEFeatures.MRAEResult, k, d))
    #     results.append(res)
    # PrintToFile(results, "Change of k (d={}) - mrDAE.txt".format(d)) 

    #- DAE vs mrDAE - Cluster Centers
    # prs = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    # ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # results = []
    # for ps in prs:
    #     res = []
    #     DAEFeatures = DAEFeature(ps)
    #     for k in ks:
    #         print("************************************************")
    #         print("  DAE(h = {}) vs mrDAE -- k = {}".format(ps, k))
    #         print("************************************************")
    #         res.append(ClassifierKMeansKNN(DAEFeatures.DAEResult, k, -1))
    #     results.append(res)
    # PrintToFile(results, "Change of k - DAE.txt")


    #- DAE-Block vs mrDAE - Cluster Centers 无监督学习
    # prs = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    # ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # results = []
    # for ps in prs:
    #     res = []
    #     Block_DAEFeatures = BlockDAEFeature(ps)
    #     for k in ks:
    #         print("************************************************")
    #         print("  DAE(h = {}) vs mrDAE -- k = {}".format(ps, k))
    #         print("************************************************")
    #         res.append(ClassifierKMeansKNN(Block_DAEFeatures.DAEResult, k, -1))
    #     results.append(res)
    # PrintToFile(results, "Change of k - Block DAE.txt")

    #- DAE-Block vs mrDAE - Cluster Centers 监督学习
    # prs = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    # results = []
    # for ps in prs:
    #     print("************************************************")
    #     print("  DAE(h = {}) vs mrDAE -- Supervised".format(ps))
    #     print("************************************************")
    #     Block_DAEFeatures = BlockDAEFeature(ps)

    #     print("********  DAE(h = {}) vs mrDAE -- MLP  ********".format(ps))
    #     results.append(ClassifierMLP(Block_DAEFeatures.DAEResult))
    #     print("********  DAE(h = {}) vs mrDAE -- SVM  ********".format(ps))
    #     results.append(ClassifierSVM(Block_DAEFeatures.DAEResult))
    # PrintToFile(results, "Change of k - Block DAE - Supervised.txt")


