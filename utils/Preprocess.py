'''
@Author: Cabrite
@Date: 2020-07-05 23:29:17
@LastEditors: Cabrite
@LastEditTime: 2020-07-17 22:12:07
@Description: Do not edit
'''

import GaborFilter, Load_Datasets, Loggers
import numpy as np
import math
import os


#@ 数据采样，单张图片全采样，用于训练分类器
def Fully_Sampling(Images, Gabor_Filter, ImageBlockSize, isWhiten, Whiten_Average=None, Whiten_U=None):
    """批量图像全采样
    
    Arguments:
        Images {np.array[numImages, rows, cols]} -- 采样图像
        Gabor_Filter {class} -- Gabor类
        ImageBlockSize {tuple} -- 采样块大小
    
    Keyword Arguments:
        Whiten_Average {np.array[rows * cols]} -- 均值矩阵 (default: {None})
        Whiten_U {np.array[rows * cols, rows * cols]} -- 旋转矩阵 (default: {None})
    
    Returns:
        np.array[numImages, numRow*numCol, block_rows*block_cols] ,
        np.array[numImages, numRow*numCol, sumGaborVisionArea] -- 全采样图像及Gabor图
    """
    block_row, block_col = ImageBlockSize
    numImages, Image_row, Image_col = Images.shape

    numRow = Image_row - block_row + 1
    numCol = Image_col - block_col + 1

    Splited_Image = np.zeros([numImages, numRow * numCol, block_row * block_col])

    for i in range(numRow):
        for j in range(numCol):
            img = Images[:, i : i + block_row, j : j + block_col]
            Splited_Image[:, i * numCol + j, :] = np.reshape(img, [numImages, block_row * block_col])
 
    if isWhiten:
        Splited_Image = np.dot((Splited_Image - Whiten_Average), Whiten_U)

    return Splited_Image


#@ 数据随机采样，特定大小图像块，用于自编码器训练
def RandomSamplingImageBlocks(Images, Images_Gabor, Gabor_Filter, ImageBlockSize, numSample, isSavingData=None):
    """随机采样图像块，用于自编码器训练
    
    Arguments:
        Images {np.array[numImages, rows, cols]} -- 需要采样的图像
        Images_Gabor {np.array[numImages, numFilters, rows, cols]} -- 需要采样图像的Gabor图
        Gabor_Filter {class} -- Gabor类
        ImageBlockSize {tuple} -- 图像块大小，必须是tuple类型
        numSample {int} -- 采样的图像块数量
    
    Keyword Arguments:
        isSavingData {string} -- 是否需要保存，需要保存图像块，则输入文件名 '*.npy' (default: {None})
    
    Returns:
        np.array -- 形状 [numSample, Block_rows * Block_cols] 的图像块数组
        np.array -- 形状 [numSample, sumGaborVisionArea] 的图像块Gabor数组
    """
    tsg = Loggers.TFprint.TFprint("Image Blocks and Gabor Images Capturing...")

    #* 参数初始化
    numImages, image_rows, image_cols = Images.shape
    block_rows, block_cols = ImageBlockSize
    half_block_rows, half_block_cols = block_rows // 2, block_cols // 2
    #* 预定义返回数组
    Image_Block = np.zeros([numSample, block_rows, block_cols])
    Image_Block_Gabor = np.zeros([numSample, Gabor_Filter.sumGaborVisionArea])

    #* 随机生成选取的图片序号
    num_CaptureImage = np.random.randint(numImages, size=[numSample])
    for index, NumberOfBlocks in enumerate(num_CaptureImage):
        #- 提取原图像
        #* 获取第n个图片及其Gabor图像
        Selected_Image = Images[NumberOfBlocks, :, :]
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

        #- 提取图像块对应的Gabor图像（感受区域）
        Selected_Image_Gabor = Images_Gabor[NumberOfBlocks, :, :, :]
        concat_result = np.array([])
        for i in range(Gabor_Filter.numGaborFilters):
            Single_Image_Block_Gabor = Selected_Image_Gabor[i, 
                                                            Selected_y - Gabor_Filter.GaborVision(i) : Selected_y + Gabor_Filter.GaborVision(i) + 1,
                                                            Selected_x - Gabor_Filter.GaborVision(i) : Selected_x + Gabor_Filter.GaborVision(i) + 1 ]
            Single_Image_Block_Gabor = np.reshape(Single_Image_Block_Gabor, Gabor_Filter.GaborVisionArea(i))
            concat_result = np.concatenate((concat_result, Single_Image_Block_Gabor))
        Image_Block_Gabor[index] = concat_result

        #- 显示日志信息
        Loggers.ProcessingBar(index + 1, numSample, CompleteLog='')

    #- 调整形状，将Image_Block 从 [numSample, block_rows, block_cols] 转成 [numSample, block_rows * block_cols]
    Image_Block = np.reshape(Image_Block, [numSample, block_rows * block_cols])

    Loggers.TFprint.TFprint("Image Blocks and Gabor Images Capturing Done!", tsg)

    if isSavingData:
        #! 保存成文件
        Loggers.PrintLog("Saving Image Blocks...")
        np.save(isSavingData[0], Image_Block)
        Loggers.PrintLog("Saving Image Blocks Done!")

        Loggers.PrintLog("Saving Gabor images...")
        np.save(isSavingData[1], Image_Block_Gabor)
        Loggers.PrintLog("Saving Gabor images Done!")
    return Image_Block, Image_Block_Gabor

def LoadRandomImageBlocks(filename):
    """读取保存的图像块数据
    
    Arguments:
        filename {[str, str]} -- 文件名， ['*.npy', '*.npy']
    
    Returns:
        np.array -- 形状 [numImages, Block_rows, Block_cols] 的图像块数组
        np.array -- 形状 [numImages, numGaborFilter, Block_rows, Block_cols] 的图像块数组
    """
    Loggers.PrintLog("Loading Image Blocks and Gabor images...")
    Image_Blocks = np.load(filename[0])
    Image_Blocks = np.reshape(Image_Blocks, [Image_Blocks.shape[0], Image_Blocks.shape[1] * Image_Blocks.shape[2]])
    Image_Blocks_Gabor = np.load(filename[1])
    Loggers.PrintLog("Loading Image Blocks and Gabor images Done!")
    return Image_Blocks, Image_Blocks_Gabor

def PCA_Whiten(Data, isWhiten):
    """数据白化
    
    Arguments:
        Data {np.array[numImages, Features]} -- 图像数据
        isWhiten {bool} -- 是否需要白化
    
    Returns:
        [np.array[numImages, Features], np.array[numImages], np.array[Features, Features]] -- 返回白化后的数据、均值及白化矩阵
    """
    tsg = Loggers.PrintLog("Now Whitening Data...")

    #- 求出均值
    Average = np.mean(Data, 0)
    Data = Data - Average

    #- 去均值后，求出协方差矩阵
    Cov = np.dot(Data.T, Data) / Data.shape[0]

    #- SVD 分解协方差矩阵
    U, S, _ = np.linalg.svd(Cov)

    #- 需要白化，则将U阵进行变换
    if isWhiten:
        U = U / np.sqrt(S + 1e-5)
    
    #- 生成白化后的矩阵
    Data_RD = np.dot(Data, U)

    Loggers.PrintLog("Whitening Data Done!", tsg)
    return Data_RD, Average, U


#@ 全图Gabor变换
def GaborAllImages(Gabor_Filter, train, batchsize=1000, method='SAME', isSavingData=None, isLogInfo=True):
    """对所有的图像进行Gabor变换
    
    Arguments:
        Gabor_Filter {class} -- Gabor类
        train {np.array} -- 训练图像
        test {np.array} -- 测试图像
    
    Keyword Arguments:
        batchsize {int} -- 每次送入GPU进行卷积的batch大小，视GPU显存大小而定 {default:{1000}}
        method {str} -- 卷积方式 (default: {'same'})
        isSavingData {str} -- 是否保存数据 (default: {None})
        isProcessingBar {bool} -- 是否需要进度条 (default: {True})
    
    Returns:
        np.array[numBlocks, numFilters, rows, cols] -- 返回值
    """
    train_gabor = Gabor_Filter.ConvoluteImages(train, batchsize=batchsize, method=method, isLogInfo=isLogInfo)
    train_gabor = train_gabor.transpose(0, 3, 1, 2)

    if isSavingData:
        #! 保存成文件
        Loggers.PrintLog("Saving Gabored Images...")
        np.save(isSavingData, train_gabor)
        Loggers.PrintLog("Saving Gabored Images Done!")

    return train_gabor

def LoadGaborImages(filename):
    """读取保存的Gabor图像块数据
    
    Arguments:
        filename {string} -- 文件名， 'train.npy'
    
    Returns:
        np.array -- 形状 [numBlocks, numFilters, Block_rows, Block_cols] 的图像块数组
    """
    Loggers.PrintLog("Loading Gabored Images...")
    train = np.load(filename)
    Loggers.PrintLog("Loading Gabored Images Done!")
    return train


#@ 同步随机采样： 选取需要采样图像及其位置，对该组图像Gabor滤波，再进行采样
def SyncSamplingImageBlocks(Images, Gabor_Filter, ImageBlockSize, numSample, SampleCapacity=5000, batchsize=1000, method='SAME', isSavingData=None, args=None):
    """同步随机采样图像块，用于自编码器训练。 将 Gabor 卷积和采样放在一起
    
    Arguments:
        Images {np.array[numImages, rows, cols]} -- 需要采样的图像
        Gabor_Filter {class} -- Gabor类
        ImageBlockSize {tuple} -- 图像块大小，必须是tuple类型
        numSample {int} -- 采样的图像块数量
    
    Keyword Arguments:
        SampleCapacity {int} -- 同步采样器一次性处理的图像数量 {default:{5000}}
        batchsize {int} -- 每次送入GPU进行卷积的batch大小，视GPU显存大小而定 {default:{1000}}
        method {str} -- 卷积方式 (default: {'same'})
        isSavingData {string} -- 是否需要保存，需要保存图像块，则输入文件名 '*.npy' (default: {None})
        args {string} -- 参数 (default: {None})
        
    Returns:
        np.array -- 形状 [numSample, Block_rows * Block_cols] 的图像块数组
        np.array -- 形状 [numSample, sumGaborVisionArea] 的图像块Gabor数组
    """
    tsg = Loggers.TFprint.TFprint("Image Blocks and Gabor Images Capturing...")

    #* 参数初始化
    numImages, image_rows, image_cols = Images.shape
    block_rows, block_cols = ImageBlockSize
    half_block_rows, half_block_cols = block_rows // 2, block_cols // 2
    #* 预定义返回数组
    Image_Block = np.zeros([numSample, block_rows, block_cols])
    Image_Block_Gabor = np.zeros([numSample, Gabor_Filter.sumGaborVisionArea])

    #* 随机生成选取的图片序号
    num_CaptureImage = np.random.randint(numImages, size=[numSample])
    total_batch = math.ceil(numSample / SampleCapacity)
    SampleIndex = 0

    #* 屏蔽 Tensorflow Debug信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    for sbatch in range(total_batch):
        #- 显示日志信息
        Loggers.ProcessingBar(sbatch + 1, total_batch, CompleteLog='')

        #- 数据处理
        #* 获取需要处理的图像
        Images_To_Capture = []
        for index in range(sbatch * SampleCapacity, min((sbatch + 1) * SampleCapacity, numSample)):
            Images_To_Capture.append(Images[num_CaptureImage[index], :, :])
        #* 获取这些图像的Gabor图像
        Images_Gabor = GaborAllImages(Gabor_Filter, Images_To_Capture, batchsize, method, isLogInfo=False)

        for Selected_Image, Selected_Image_Gabor in zip(Images_To_Capture, Images_Gabor):
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
            Image_Block[SampleIndex] = Selected_Image[Selected_y - half_block_rows : Selected_y + half_block_rows + 1, Selected_x - half_block_cols: Selected_x + half_block_cols + 1]

            #- 提取图像块对应的Gabor图像（感受区域）
            concat_result = np.array([])
            for i in range(Gabor_Filter.numGaborFilters):
                Single_Image_Block_Gabor = Selected_Image_Gabor[i, 
                                                                Selected_y - Gabor_Filter.GaborVision(i) : Selected_y + Gabor_Filter.GaborVision(i) + 1,
                                                                Selected_x - Gabor_Filter.GaborVision(i) : Selected_x + Gabor_Filter.GaborVision(i) + 1 ]
                Single_Image_Block_Gabor = np.reshape(Single_Image_Block_Gabor, Gabor_Filter.GaborVisionArea(i))
                concat_result = np.concatenate((concat_result, Single_Image_Block_Gabor))
            Image_Block_Gabor[SampleIndex] = concat_result
            SampleIndex += 1

    #- 调整形状，将Image_Block 从 [numSample, block_rows, block_cols] 转成 [numSample, block_rows * block_cols]
    Image_Block = np.reshape(Image_Block, [numSample, block_rows * block_cols])

    Loggers.TFprint.TFprint("Image Blocks and Gabor Images Capturing Done!", tsg)

    if isSavingData:
        #! 保存成文件
        Loggers.PrintLog("Saving Image Blocks...")
        np.save(isSavingData[0], Image_Block)
        Loggers.PrintLog("Saving Image Blocks Done!")

        Loggers.PrintLog("Saving Gabor images...")
        np.save(isSavingData[1], Image_Block_Gabor)
        Loggers.PrintLog("Saving Gabor images Done!")
    
    if args:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.TFInfo)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    return Image_Block, Image_Block_Gabor


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #- 读取数据集
    Train_X, Train_Y, Test_X, Test_Y = Load_Datasets.Preprocess_Raw_Data("./Datasets", "MNIST", True, True)

    #- 生成Gabor滤波器
    ksize = (29, 29)
    Lambda = [1, 2, 3, 4]
    numTheta = 8
    Theta = [np.pi / numTheta * i for i in range(numTheta)]
    Beta = [1]
    Gamma = [0.5, 1]
    RI_Part = 'b'

    GaborClass = GaborFilter.Gabor()
    GaborClass.setParam(ksize, Theta, Lambda, Gamma, Beta, RI_Part)


    #- 图像块采样
    savefile_ImageBlocks = ['ImageBlocks.npy', 'ImageBlocksGabor.npy']
    ImageBlockSize = (11, 11)
    numSample = 400
    
    #- 对图像进行Gabor变换
    # Train_X_Gabor = GaborAllImages(GaborClass, Train_X, batchsize=5000)
    # Test_X_Gabor = GaborAllImages(GaborClass, Test_X, batchsize=5000)
    #* 自动采样图像块
    # Image_Blocks, Image_Blocks_Gabor = RandomSamplingImageBlocks(Train_X, Train_X_Gabor, GaborClass, ImageBlockSize, numSample, isSavingData=savefile_ImageBlocks, isLog=1000)
    # Image_Blocks, Image_Blocks_Gabor = RandomSamplingImageBlocks(Train_X, Train_X_Gabor, GaborClass, ImageBlockSize, numSample)

    #* 同步采样图像块
    Image_Blocks, Image_Blocks_Gabor = SyncSamplingImageBlocks(Train_X, GaborClass, ImageBlockSize, numSample, 33)

    #* 读取已保存的图像块
    # Image_Blocks, Image_Blocks_Gabor = LoadRandomImageBlocks(savefile_ImageBlocks)

    #* 数据白化
    Image_Blocks, Whiten_Avg, Whiten_U = PCA_Whiten(Image_Blocks, True)