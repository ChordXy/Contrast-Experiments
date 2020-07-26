'''
@Author: Cabrite
@Date: 2020-07-02 21:34:36
@LastEditors: Cabrite
@LastEditTime: 2020-07-25 23:10:48
@Description: 读取数据集
'''

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import Loggers
import struct
import gzip
import os


#@ 读取数据集
def Load_MNIST_Like_Dataset(Dataset_folder):
    """读取 MNIST、Fashion-MNIST 库
    
    Arguments:
        Dataset_folder {string} -- 路径
    
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

def Load_SVHN_Dataset(Dataset_folder):
    """读取SVHN数据

    Args:
        Dataset_folder (str): 路径
    """
    def load_svhn_data(Mat_file):
        data = sio.loadmat(Mat_file)
        x_data = np.transpose(data['X'], [3, 0, 1, 2])
        y_data = data['y']
        x_data_gray = np.round(np.dot(x_data, [0.299, 0.587, 0.114])).astype(np.uint8)
        return x_data_gray, y_data

    x_train, y_train = load_svhn_data(os.path.join(Dataset_folder, 'train_32x32.mat'))
    x_test, y_test = load_svhn_data(os.path.join(Dataset_folder, 'test_32x32.mat'))

    #- 数据集中， 数字 0 的标签是 10， 需要转换回0
    y_train_refined = [elem if elem < 10 else elem - 10 for elem in y_train]
    y_test_refined = [elem if elem < 10 else elem - 10 for elem in y_test]
    
    return x_train, y_train_refined, x_test, y_test

def Preprocess_Raw_Data(dataset_root, Data_Name="", one_hot=False, normalization=False):
    """数据预处理
    
    Arguments:
        dataset_root {string} -- 数据路径
    
    Keyword Arguments:
        mode {int} -- 模式，0：mnist, Fashion-mnist  1：SVHN (default: {0})
        one_hot {bool} -- 是否进行One_Hot编码 (default: {False})
        normalization {bool} -- 是否归一化数据 (default: {False})
    
    Returns:
        np.array, np.array, np.array, np.array -- 预处理后的数据集
    """
    Loggers.TFprint.TFprint("Loading {} Data...".format(Data_Name))
    
    dataset_directory = os.path.join(dataset_root, Data_Name)
    #* 自定读取
    if Data_Name == "SVHN":
        Train_X, Train_Y, Test_X, Test_Y = Load_SVHN_Dataset(dataset_directory)
    else:
        Train_X, Train_Y, Test_X, Test_Y = Load_MNIST_Like_Dataset(dataset_directory)
    
    if one_hot:
        Train_Y = np.array([[1 if i==elem else 0 for i in range(10)] for elem in Train_Y], dtype=np.float32)
        Test_Y = np.array([[1 if i==elem else 0 for i in range(10)] for elem in Test_Y], dtype=np.float32)
    
    if normalization:
        Train_X = Train_X / 255
        Test_X = Test_X / 255
    Loggers.TFprint.TFprint("Loading {} Data Done!".format(Data_Name))
    return Train_X, Train_Y, Test_X, Test_Y

#@ 附加函数
def DisplayDatasets(images, labels=None, figure_row=8, figure_col=8, cmap='gray'):
    """显示数据集图像
    
    Arguments:
        images {np.array [num, rows, cols]} -- 图像
    
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
                label = labels[image_count]
                if len(label) == 1:
                    title = str(label[0])
                else:
                    title = str(np.argmax(label))
                plt.title(title)
                image_count += 1
                
                #! 关闭坐标轴
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Train_X, Train_Y, Test_X, Test_Y = Preprocess_Raw_Data("./Datasets", "Fashion_MNIST", True, True)
    # DisplayDatasets(Train_X[0:64])


    Train_X, Train_Y, Test_X, Test_Y = Preprocess_Raw_Data("./Datasets", "SVHN", True, True)
    DisplayDatasets(Test_X[0:64], Test_Y[0:64])
