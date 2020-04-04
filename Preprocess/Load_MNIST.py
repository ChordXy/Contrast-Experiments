'''
@Description: 读取MNIST数据集
@Author: Cabrite
@Date: 2019-12-13 14:30:24
@LastEditors  : Cabrite
@LastEditTime : 2020-01-04 15:03:43
'''

import os
import gzip
import struct
import datetime
import numpy as np
import matplotlib.pyplot as plt

#@ 读取数据集
def load_Mnist_Dataset(Dataset_folder):
    """读取MNIST手写字体库
    
    Arguments:
        Dataset_folder {string} -- MNIST路径
    
    Returns:
        np.array, np.array, np.array, np.array -- 读取的数据集
    """
    image_files = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    label_files = ['train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    image_paths = [os.path.join(Dataset_folder, file) for file in image_files]
    label_paths = [os.path.join(Dataset_folder, file) for file in label_files]

    with gzip.open(label_paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(image_paths[0], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(label_paths[1], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(image_paths[1], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    
    return x_train, y_train, x_test, y_test

def Preprocess_MNIST_Data(dataset_root, one_hot=False, normalization=False):
    """数据预处理
    
    Arguments:
        dataset_root {string} -- MNIST数据路径
    
    Keyword Arguments:
        one_hot {bool} -- 是否进行One_Hot编码 (default: {False})
        normalization {bool} -- 是否归一化数据 (default: {False})
    
    Returns:
        np.array, np.array, np.array, np.array -- 预处理后的数据集
    """
    PrintLog("Loading MNIST Data...")
    
    #* TensorFlow读取方式
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets(dataset_root, one_hot=True)

    #* 自定读取
    Train_X, Train_Y, Test_X, Test_Y = load_Mnist_Dataset(dataset_root)
    
    if one_hot:
        Train_Y = np.array([[1 if i==elem else 0 for i in range(10)] for elem in Train_Y], dtype=np.float32)
        Test_Y = np.array([[1 if i==elem else 0 for i in range(10)] for elem in Test_Y], dtype=np.float32)
    
    if normalization:
        Train_X = Train_X / 255
        Test_X = Test_X / 255
    PrintLog("Loading MNIST Data Done!")
    return Train_X, Train_Y, Test_X, Test_Y

#@ 附加函数
def DisplayMNIST(images, figure_row=8, figure_col=8, cmap='gray'):
    """显示MNIST图像
    
    Arguments:
        images {np.array} -- 图像
    
    Keyword Arguments:
        figure_row {int} -- [每一行显示的图像对数] (default: {8})
        figure_col {int} -- [列数] (default: {8})
        cmap {str} -- [灰度图] (default: {'gray'})
    """
    figure_size = figure_row * figure_col
    numImages = images.shape[0]
    numFigure = int(numImages / figure_size) + 1
    image_count = 0
    Done_flag = False

    for figure_NO in range(numFigure):
        #! 防止出现空白的 figure
        if Done_flag == True or image_count == numImages:
            break
        #* 绘制新的 figure
        plt.figure(figure_NO)
        for i in range(figure_row):
            if Done_flag == True:
                break
            for j in range(figure_col):
                if image_count == numImages:
                    Done_flag = True
                    break

                plt.subplot(figure_row, figure_col, i * figure_col + j + 1)
                plt.imshow(images[image_count], cmap=cmap)
                image_count += 1
                
                #! 关闭坐标轴
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
    plt.show()

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

if __name__ == "__main__":
    Train_X, Train_Y, Test_X, Test_Y = Preprocess_MNIST_Data("./Datasets/MNIST_Data", True, True)
    DisplayMNIST(Train_X[0:75])
