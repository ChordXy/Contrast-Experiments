'''
@Author: Cabrite
@Date: 2020-07-05 23:51:08
@LastEditors: Cabrite
@LastEditTime: 2020-07-25 15:31:03
@Description: Do not edit
'''

from tensorflow.contrib.layers import xavier_initializer
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import Preprocess
import Loggers
import math
import os


class MultiResolutionDAE():
    def __init__(self):
        self.isLoadModel = True
        self.set_AE_Parameters()
        self.set_AE_Training_Parameters()
        self.set_TiedAE_Training_Parameters()

    #- 初始化 AE 参数
    def Init_DAE(self, Gabor_Filter, Block=(11, 11)):
        self.Gabor_Filter = Gabor_Filter
        self.sumGaborVisionArea = self.Gabor_Filter.sumGaborVisionArea
        self.ImageBlockSize = Block
        self.numPixels = Block[0] * Block[1]

    def set_AE_Input_Data(self, Image_Blocks, Image_Blocks_Gabor):
        self.Main_Inputs = Image_Blocks
        self.Siamese_Inputs = Image_Blocks_Gabor
        self.numSamples = Image_Blocks.shape[0]

    def set_AE_Parameters(self, n_Hiddens=1024, reconstruction_reg=0.5, measurement_reg=0.1, sparse_reg=0.1, gaussian=0.02, batch_size=500, display_step=1):
        #* 隐含层神经元数量
        self.n_Hiddens = n_Hiddens
        #* 重建比重
        self.reconstruction_reg = reconstruction_reg
        #* 相似度量比重
        self.measurement_reg = measurement_reg
        #* 稀疏性比重
        self.sparse_reg = sparse_reg
        #* 高斯噪声
        self.gaussian = gaussian
        #* AE训练 batch 量
        self.batch_size = batch_size
        #* Log显示次数
        self.display_step = display_step
        self.model_path = './Model_mrDAE/mrDAE.ckpt'

    def set_AE_Training_Parameters(self, epochs=500, lr_init=2e-1, lr_decay_step=4, lr_decay_rate=0.98):
        self.AE_Training_Epochs = epochs
        self.AE_Learning_Rate_Init = lr_init
        self.AE_Learning_Rate_Decay_Step = lr_decay_step
        self.AE_Learning_Rate_Decay_Rate = lr_decay_rate

    def set_TiedAE_Training_Parameters(self, epochs=500, lr_init=2e-1, lr_decay_step=4, lr_decay_rate=0.98):
        self.TiedAE_Training_Epochs = epochs
        self.TiedAE_Learning_Rate_Init = lr_init
        self.TiedAE_Learning_Rate_Decay_Step = lr_decay_step
        self.TiedAE_Learning_Rate_Decay_Rate = lr_decay_rate

    #- AE 公有
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
        siamese_layer_bn = tf.contrib.layers.batch_norm(siamese_layer, 0.9, epsilon = 1e-5, trainable=isTrainable)
        return siamese_layer_bn

    #- Tied AE网络
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

    def Build_TiedAutoEncoderNetwork(self):
        """带Siamese旁支、绑定权重的AE网络
        """
        ############################  重置网络  ############################
        tf.reset_default_graph()
    
        ############################  设置衰减学习率  ############################
        self.isLoadModel = False
        total_batch = math.ceil(self.numSamples / self.batch_size)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay( learning_rate=self.TiedAE_Learning_Rate_Init, 
                                                    global_step=global_step, 
                                                    decay_steps=total_batch * self.TiedAE_Learning_Rate_Decay_Step, 
                                                    decay_rate=self.TiedAE_Learning_Rate_Decay_Rate)

        ############################  初始化网络输入  ############################
        input_Main = tf.placeholder(tf.float32, [None, self.numPixels])
        input_Siamese = tf.placeholder(tf.float32, [None, self.sumGaborVisionArea])

        ############################  构建网络  ############################
        encoder_layer, decoder_layer = self.TiedEncoderDecoderLayer(input_Main, self.numPixels, self.n_Hiddens, tf.nn.leaky_relu)
        siamese_layer = self.SiameseLayer(input_Siamese, self.sumGaborVisionArea, self.n_Hiddens, tf.nn.leaky_relu)

        #* 重建损失
        loss_reconstruction = tf.reduce_mean(tf.pow(tf.subtract(input_Main, decoder_layer), 2.0))
        #* 度量损失
        loss_measurement = tf.reduce_mean(tf.abs(tf.subtract(encoder_layer, siamese_layer)))
        #* 稀疏性
        loss_sparse = tf.reduce_mean(tf.abs(siamese_layer))
        #* 最终损失
        loss = self.reconstruction_reg * loss_reconstruction + self.measurement_reg * loss_measurement + self.sparse_reg * loss_sparse
        #* 优化函数
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ############################  网络保存  ############################
        saver = tf.train.Saver()

        ############################  训练网络  ############################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.TiedAE_Training_Epochs):
                avg_loss = 0
                for i in range(total_batch):
                    #* 提取每个Batch对应的数据
                    batch_main = self.Main_Inputs[i * self.batch_size : min((i + 1) * self.batch_size, self.numSamples), :]
                    batch_siamese = self.Siamese_Inputs[i * self.batch_size : min((i + 1) * self.batch_size, self.numSamples), :]
                    #* 加入噪声
                    batch_main_noise = batch_main + self.gaussian * np.random.randn(*batch_main.shape)
                    #* 训练网络
                    _, ls = sess.run([optimizer, loss], feed_dict={input_Main : batch_main_noise, input_Siamese : batch_siamese})
                    avg_loss += ls / total_batch
                    Loggers.ProcessingBar(i + 1, total_batch, isClear=True)
                    
                if (epoch + 1) % self.display_step == 0:
                    message = "Epoch : " + '%04d' % (epoch + 1) + " loss = " + "{:.9f}".format(avg_loss)
                    Loggers.TFprint.TFprint(message)
                    save_path = saver.save(sess, self.model_path, global_step = epoch)

            Loggers.TFprint.TFprint("Finished!")
            save_path = saver.save(sess, self.model_path)
            Loggers.TFprint.TFprint("Model saved in file : " + save_path)


    #- AE网络 （暂时不用）
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
    
    def Build_AutoEncoderNetwork(self):
        """带Siamese旁支的AE网络
        """
        ############################  重置网络  ############################
        tf.reset_default_graph()

        ############################  设置衰减学习率  ############################
        self.isLoadModel = False
        total_batch = math.ceil(self.numSamples / self.batch_size)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay( learning_rate=self.AE_Learning_Rate_Init, 
                                                    global_step=global_step, 
                                                    decay_steps=total_batch * self.AE_Learning_Rate_Decay_Step, 
                                                    decay_rate=self.AE_Learning_Rate_Decay_Rate)

        ############################  初始化网络输入  ############################
        input_Main = tf.placeholder(tf.float32, [None, self.numPixels])
        input_Siamese = tf.placeholder(tf.float32, [None, self.sumGaborVisionArea])
 
        ############################  构建网络  ############################
        encoder_layer = self.EncoderLayer(input_Main, self.numPixels, self.n_Hiddens, tf.nn.leaky_relu)
        decoder_layer = self.DecoderLayer(encoder_layer, self.n_Hiddens, self.numPixels, tf.nn.leaky_relu)
        siamese_layer = self.SiameseLayer(input_Siamese, self.sumGaborVisionArea, self.n_Hiddens, tf.nn.leaky_relu)

        #* 重建损失
        loss_reconstruction = tf.reduce_mean(tf.pow(tf.subtract(input_Main, decoder_layer), 2.0))
        #* 度量损失
        loss_measurement = tf.reduce_mean(tf.abs(tf.subtract(encoder_layer, siamese_layer)))
        #* 稀疏性
        loss_sparse = tf.reduce_mean(tf.abs(siamese_layer))
        #* 最终损失
        loss = self.reconstruction_reg * loss_reconstruction + self.measurement_reg * loss_measurement + self.sparse_reg * loss_sparse
        #* 优化函数
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ############################  初始化参数  ############################
        saver = tf.train.Saver()

        ############################  训练网络  ############################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.AE_Training_Epochs):
                avg_loss = 0
                for i in range(total_batch):
                    #* 提取每个Batch对应的数据
                    batch_main = self.Main_Inputs[i * self.batch_size : min((i + 1) * self.batch_size, self.numSamples), :]
                    batch_siamese = self.Siamese_Inputs[i * self.batch_size : min((i + 1) * self.batch_size, self.numSamples), :]
                    #* 加入噪声
                    batch_main_noise = batch_main + self.gaussian * np.random.randn(*batch_main.shape)
                    #* 训练网络
                    _, ls = sess.run([optimizer, loss], feed_dict={input_Main : batch_main_noise, input_Siamese : batch_siamese})
                    avg_loss += ls / total_batch
                    Loggers.ProcessingBar(i + 1, total_batch, isClear=True)
                    
                if (epoch + 1) % self.display_step == 0:
                    message = "Epoch : " + '%04d' % (epoch + 1) + " loss = " + "{:.9f}".format(avg_loss)
                    Loggers.TFprint.TFprint(message)
                    save_path = saver.save(sess, self.model_path, global_step = epoch)

            Loggers.TFprint.TFprint("Finished!")
            save_path = saver.save(sess, self.model_path)
            Loggers.TFprint.TFprint("Model saved in file : " + save_path)


    #- AE 编码，提取 mrDAE 特征
    def Encodering_Feature(self, Data, isWhiten, Whiten_Average=None, Whiten_U=None, batch_size=1000):
        """提取mrDAE编码特征

        Args:
            Data ([num, rows, cols]): 需要编码的数据

        Returns:
            [num, nrows*ncols, n_hiddens]: 编码后的特征
        """
        #- 初始化参数
        block_row, block_col = self.ImageBlockSize
        numData, Data_row, Data_col = Data.shape
        numRow = Data_row - block_row + 1
        numCol = Data_col - block_col + 1
        ksize = [1, numRow // 2, numCol // 2, 1]
        stride = [1, numRow - numRow // 2, numCol - numCol //2, 1]

        totalbatch = math.ceil(numData / batch_size)

        #! 演示分割结果
        # display_no = 0
        # DisplaySplitResult(Data[display_no], np.reshape(Splited_Data[display_no, :, :], [numRow * numCol, block_row, block_col]), numRow, numCol)
        
        #- 提取高维特征并池化，重新组合
        tf.reset_default_graph()

        Encodered_Data = np.zeros([numData, 4 * self.n_Hiddens])

        #* 输入
        input_Main = tf.placeholder(tf.float32, [None, numRow * numCol, self.numPixels])
        input_Size = tf.placeholder(tf.int32)
        #* 改变形状，适应批量乘法
        input_Main_Reshaped = tf.reshape(input_Main, [input_Size * numRow * numCol, self.numPixels])
        #* 特征编码
        encoder_layer, _ = self.TiedEncoderDecoderLayer(input_Main_Reshaped, self.numPixels, self.n_Hiddens, tf.nn.leaky_relu, False)
        #* 变换形状
        encoder_Reshaped = tf.reshape(encoder_layer, [input_Size, numRow, numCol, self.n_Hiddens])
        #* 均值池化
        avgpool = tf.nn.avg_pool(encoder_Reshaped, ksize, stride, 'VALID')
        #* 变换形状，首尾拼接
        avgpool_Reshaped = tf.reshape(avgpool, [input_Size, 4 * self.n_Hiddens])

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.model_path)

            tsg = Loggers.TFprint.TFprint("Extracting High Dimensional Features...")
            for i in range(totalbatch):
                Splited_Image = Preprocess.Fully_Sampling(  Data[i * batch_size : min((i + 1) * batch_size, numData)],
                                                            self.Gabor_Filter,
                                                            self.ImageBlockSize,
                                                            isWhiten=isWhiten,
                                                            Whiten_Average=Whiten_Average,
                                                            Whiten_U=Whiten_U)
                Encodered_Data[i * batch_size : min((i + 1) * batch_size, numData), :] = sess.run(avgpool_Reshaped, feed_dict = {input_Main : Splited_Image, input_Size : Splited_Image.shape[0]})
                Loggers.ProcessingBar(i + 1, totalbatch, CompleteLog='')
            Loggers.TFprint.TFprint("Extracting High Dimensional Features Done!", tsg)
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

    def get_mrDAE_Train_Test_Feature(self, Train, Test, isWhiten, Whiten_Average=None, Whiten_U=None, isFeatureReduction=True, batch_size=1000):
        """获取mrDAE对训练及测试数据的编码特征

        Args:
            Train ([num, numPixels]): 输入数据
            Test ([num, numPixels]): 测试数据
            Whiten_Average ([array], optional): 白化均值. Defaults to None.
            Whiten_U ([array], optional): 白化矩阵. Defaults to None.

        Returns:
            [array, array]: 特征编码并筛选后的结果
        """
        if self.isLoadModel:
            Whiten_Average = np.load('./Model_mrDAE/Whiten_Average.npy')
            Whiten_U = np.load('./Model_mrDAE/Whiten_MatrixU.npy')

        Train_feature = self.Encodering_Feature(Train, isWhiten, Whiten_Average, Whiten_U, batch_size)
        Test_feature = self.Encodering_Feature(Test, isWhiten, Whiten_Average, Whiten_U, batch_size)
        if isFeatureReduction:
            Train_feature, Test_feature = self.FeatureReduction(Train_feature, Test_feature)
        return Train_feature, Test_feature


    #- 可视化
    def Display_Reconstruction(self, numImages, Gabor_Filter):
        """显示重建结果
        
        Arguments:
            numImages {int} -- 重建的图像数量
        """
        tf.reset_default_graph()

        ############################  构建网络  ############################
        input_Main = tf.placeholder(tf.float32, [None, self.numPixels])
        encoder_layer = self.EncoderLayer(input_Main, self.numPixels, self.n_Hiddens, tf.nn.leaky_relu)
        decoder_layer = self.DecoderLayer(encoder_layer, self.n_Hiddens, self.numPixels, tf.nn.leaky_relu)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.model_path)

            batch_xs = self.Main_Inputs[0 : numImages]
            output_val = sess.run([decoder_layer], feed_dict = {input_Main : batch_xs})
            
            batch_xs = np.reshape(batch_xs, [numImages, *self.ImageBlockSize])
            output_val = np.reshape(output_val, [numImages, *self.ImageBlockSize])

            DisplayReconstructionResult(batch_xs, output_val)

    def Visualization(self, num_Max_Display=1, num_Min_Display=1):
        """权重可视化

        Args:
            num_Max_Display (int, optional): 最大权重显示图像数. Defaults to 1.
            num_Min_Display (int, optional): 最小权重显示图像数. Defaults to 1.
        """
        tf.reset_default_graph()

        siamese_weight = tf.get_variable('Siamese_Layer/Siamese_Weight', [self.sumGaborVisionArea, self.n_Hiddens])

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #* 载入模型
            saver.restore(sess, self.model_path)
            #* 载入旁支权重
            weight = sess.run(siamese_weight)
        
        #* 交换轴：由(inputs, hiddens) -> (hiddens, inputs)
        weight = weight.transpose(1, 0)
        #* 统计权重
        sum_weight = weight.sum(1)
        #* 权重排序
        order_weight = np.argsort(sum_weight)
            
        #* 截取指定数据
        max_show_sequence = order_weight[- num_Max_Display : ]
        min_show_sequence = order_weight[0 : num_Min_Display]
        max_show_weight = weight[max_show_sequence]
        min_show_weight = weight[min_show_sequence]
        
        #* 显示图像
        # 最大
        for index, single_weight in enumerate(max_show_weight):
            self.DisplayWeight(single_weight, "Max - {} Weight".format(len(max_show_sequence) - index), 8, 4)
        # 最小
        for index, single_weight in enumerate(min_show_weight):
            self.DisplayWeight(single_weight, "Min - {} Weight".format(index + 1), 8, 4)    

    def DisplayWeight(self, Data, Display_Name, figure_row=None, figure_col=None):
        """显示权重

        Args:
            Data ([type]): 权重
            Display_Name ([type]): 名称
        """
        counting = 0
        #* 每个不同大小感受野的数据
        Images = []
        #* 感受野大小
        block_len = []
        for i in range(self.Gabor_Filter.numGaborFilters):
            #* 提取感受野面积
            block_area = self.Gabor_Filter.GaborVisionArea(i)
            #* 按顺序提取权重
            Images.append(Data[counting : counting + block_area])
            #* 存入感受野边长
            block_len.append(int(math.sqrt(block_area)))
            #* 长度计量
            counting += block_area
        
        #* 感受野大小集合
        Images_Status = list(set(block_len))
        for index, elem in enumerate(Images_Status):
            #* 新建图像窗口
            plt.figure(Display_Name + ' Block {} * {}'.format(elem, elem))
            #* 找到当前感受野大小的索引
            all_same_elem_index = [index for index in range(len(block_len)) if block_len[index] == elem]

            #* 在Images感受野权重中，提取出特定大小感受野的权重
            display_data = []
            for edx in all_same_elem_index:
                display_data.append(Images[edx])
            #* 总共需要显示的感受野图像数
            figure_num = len(display_data)
            #* 显示的图像横纵方向个数
            if not (figure_col or figure_row):
                figure_col = math.ceil(math.sqrt(figure_num))
                figure_row = math.ceil(math.sqrt(figure_num))
            
            if elem == 1:
                display_data = np.reshape(display_data, [figure_num])
                plt.bar(range(figure_num), display_data, width=0.5)
            else:
                for i in range(figure_num):
                    ax = plt.subplot(figure_row, figure_col, i + 1)
                    #* 图像重新调整大小
                    Image = np.reshape(display_data[i], [elem, elem])
                    # ax.set_title("")
                    plt.imshow(Image)
                    
                    #! 关闭坐标轴
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                        
        plt.show()


class mrDAE_Classifier():
    def __init__(self):
        self.set_MLP_Training_Parameters()

    #- 初始化 MLP 参数
    def set_MLP_Input_Data(self, Train_X, Train_Y, Test_X, Test_Y):
        self.Train_X = Train_X
        self.Train_Y = Train_Y
        self.Test_X = Test_X 
        self.Test_Y = Test_Y
        self.MLP_numTrain = self.Train_X.shape[0]
        self.n_features = self.Train_X.shape[1]
        self.n_class = Train_Y.shape[1]

    def set_MLP_Training_Parameters(self, n_Hiddens=2048, epochs=500, batchsize=200, gaussian=0.02, lr_init=2e-1, lr_step=4, lr_rate=0.98, display_step=1, dropout=1):
        self.n_Hiddens = n_Hiddens
        self.MLP_Training_Epochs = epochs
        self.MLP_Batch_Size = batchsize

        self.MLP_gaussian = gaussian
        self.MLP_Learning_Rate_Init = lr_init
        self.MLP_Learning_Rate_Steps = lr_step
        self.MLP_Learning_Rate_Rates = lr_rate
        self.display_step = display_step
        self.DropOut=dropout

        self.model_path = './Model_mrDAE_Classifier/mrDAE_Classification.ckpt'
        self.max_keep_model_path = './Model_mrDAE_Classifier/mrDAE_Classification_Max.ckpt'

    def set_MLP_Test_Data(self, Test_X, Test_Y):
        self.Test_X = Test_X 
        self.Test_Y = Test_Y
        self.n_features = self.Test_X.shape[1]
        self.n_class = Test_Y.shape[1]

    #- MLP 分类网络
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

    def SoftmaxClassifyLayer(self, Input_Layer, Input_Size, Output_Size, isTrainable=True, isTraining=True):
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
        layer_out = tf.matmul(Input_Layer, weight) + bias
        bn_layer = tf.contrib.layers.batch_norm(layer_out, 0.9, epsilon = 1e-5, is_training=isTraining)
        softmax_layer = tf.nn.softmax(bn_layer)
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

    def Build_ClassificationNetwork(self):
        """建立分类网络
        """
        ############################  重置网络  ############################
        tf.reset_default_graph()

        ############################  初始化网络输入  ############################
        input_Feature = tf.placeholder(tf.float32, [None, self.n_features])
        y = tf.placeholder(tf.float32, [None, self.n_class])
        dropout_keep_prob = tf.placeholder("float")
        flag_training = tf.placeholder(dtype=tf.bool)

        ############################  初始化批次及学习率  ############################
        totalbatch = math.ceil(self.MLP_numTrain / self.MLP_Batch_Size)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay( learning_rate=self.MLP_Learning_Rate_Init, 
                                                    global_step=global_step, 
                                                    decay_steps=totalbatch * self.MLP_Learning_Rate_Steps, 
                                                    decay_rate=self.MLP_Learning_Rate_Rates)
 
        ############################  构建网络  ############################
        #* 隐含层
        hidden_layer = self.HiddenFullyConnectedLayer(input_Feature, self.n_features, self.n_Hiddens, tf.nn.leaky_relu, dropout_keep_prob)
        #* 分类层
        pred = self.SoftmaxClassifyLayer(hidden_layer, self.n_Hiddens, self.n_class, True, flag_training)
        # pred = self.MLPLayer(hidden_layer, self.n_Hiddens, self.n_class)

        #* 最终损失
        loss = tf.reduce_mean(- tf.reduce_sum(y * tf.log(pred + 1e-6), reduction_indices = 1))

        #* 优化函数
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ############################  准确率  ############################
        correction_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

        ############################  初始化参数  ############################
        saver = tf.train.Saver()
        Max_Accuracy = 0
        
        ############################  训练网络  ############################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.MLP_Training_Epochs):
                avg_loss = 0
                for i in range(totalbatch):
                    #* 提取每个Batch对应的数据
                    batch_xs = self.Train_X[i * self.MLP_Batch_Size : min((i + 1) * self.MLP_Batch_Size, self.MLP_numTrain), :]
                    batch_ys = self.Train_Y[i * self.MLP_Batch_Size : min((i + 1) * self.MLP_Batch_Size, self.MLP_numTrain), :]
                    #* 加入噪声
                    batch_xs_noise = batch_xs + self.MLP_gaussian * np.random.randn(*batch_xs.shape)
                    #* 训练网络
                    _, ls = sess.run([optimizer, loss], feed_dict={input_Feature : batch_xs_noise, y : batch_ys, dropout_keep_prob : self.DropOut, flag_training : True})
                    avg_loss += ls / totalbatch
                    Loggers.ProcessingBar(i + 1, totalbatch, isClear=True)
                
                f_acc = accuracy.eval(feed_dict = {input_Feature : self.Test_X, y : self.Test_Y, dropout_keep_prob : 1., flag_training : False})
                #* 保存最大准确率模型
                if f_acc > Max_Accuracy:
                    saver.save(sess, self.max_keep_model_path)
                    Max_Accuracy = f_acc

                if (epoch + 1) % self.display_step == 0:
                    learn_rate = sess.run(learning_rate)
                    message = "Epoch : " + '%04d' % (epoch + 1) + \
                            " Loss = " + "{:.5f}".format(avg_loss) + \
                            " Learning Rate = " + "{:.5f}".format(learn_rate) + \
                            " Final Accuracy = " + "{:.9f}".format(f_acc)
                    Loggers.TFprint.TFprint(message)
                    save_path = saver.save(sess, self.model_path, global_step = epoch)

            Loggers.TFprint.TFprint("Finished!")
            save_path = saver.save(sess, self.model_path)

            #* 测试集结果
            Loggers.TFprint.TFprint("Max Testing Accuracy : {}".format(Max_Accuracy))

    def DisplayResult(self):
        """显示分类网络结果
        """
        ############################  重置网络  ############################
        tf.reset_default_graph()

        ############################  初始化网络输入  ############################
        input_Feature = tf.placeholder(tf.float32, [None, self.n_features])
        y = tf.placeholder(tf.float32, [None, self.n_class])
        dropout_keep_prob = tf.placeholder("float")
        flag_training = tf.placeholder(dtype=tf.bool)

        ############################  构建网络  ############################
        #* 隐含层
        hidden_layer = self.HiddenFullyConnectedLayer(input_Feature, self.n_features, self.n_Hiddens, tf.nn.leaky_relu, dropout_keep_prob)
        #* 分类层
        pred = self.SoftmaxClassifyLayer(hidden_layer, self.n_Hiddens, self.n_class, True, flag_training)

        ############################  准确率  ############################
        correction_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

        ############################  初始化参数  ############################
        saver = tf.train.Saver()

        prediction = tf.argmax(pred, 1)
        
        ############################  训练网络  ############################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.model_path)

            Loggers.TFprint.TFprint("Display Results!")
            #* 显示准确率
            f_acc, prediction_value = sess.run([accuracy, prediction], feed_dict = {input_Feature : self.Test_X, y : self.Test_Y, dropout_keep_prob : 1., flag_training : False})
            Loggers.TFprint.TFprint("Accuracy = {}".format(f_acc))

            #* 获取混淆矩阵
            real_value = [np.argmax(elem) for elem in self.Test_Y]
            Confusion = np.zeros([self.n_class, self.n_class], dtype=np.int32)

            for rv, pv in zip(real_value, prediction_value):
                Confusion[rv][pv] += 1
            
            print(Confusion)
            print(Confusion.shape)

            #* 绘制混淆图
            ticks = range(self.n_class)
            tick_names = range(self.n_class)


            plt.imshow(np.array(Confusion), cmap=plt.cm.Blues)
            
            plt.title('Confusion Matrix')
            plt.xlabel('Prediction')
            plt.ylabel('Reality')
            plt.xticks(ticks, tick_names)
            plt.yticks(ticks, tick_names)
            plt.colorbar()

            

            for rv in range(self.n_class):
                for pv in range(self.n_class):
                    plt.text(rv, pv, confusion[rv][pv])

            plt.savefig('Confusion Matrix.jpg')

            Loggers.TFprint.TFprint("Finished!")


#@ 附加函数
def DisplayReconstructionResult(origin, reconstruction, figure_row=4, figure_col=8):
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

def DisplaySplitResult(origin_image, split_image, rows, cols):
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


if __name__ == "__main__":
    pass
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
