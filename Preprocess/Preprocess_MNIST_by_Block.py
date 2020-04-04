'''
@Description: 数据预处理-提取图像块，Gabor变换。
@Author: Cabrite
@Date: 2019-12-13 16:20:36
@LastEditors  : Cabrite
@LastEditTime : 2020-01-04 15:04:39
'''

#- 该方法采用的是先取图像块，再做Gabor变换

import datetime
import numpy as np
import GaborFilter, Load_MNIST


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

def SamplingImageBlocks(Images, ImageBlockSize, numSample, whiten=True, isSavingData=None, isLog=0):
    """采样图像块
    
    Arguments:
        Images {np.array} -- 需要采样的图像
        ImageBlockSize {tuple} -- 图像块大小，必须是tuple类型
        numSample {int} -- 采样的图像块数量
    
    Keyword Arguments:
        whiten {bool} -- 是否白化数据 (default: {True})
        isSavingData {string} -- 是否需要保存，需要保存图像块，则输入文件名 '*.npy' (default: {None})
        isLog {int} -- 是否需要打log {default:{0}}
    
    Returns:
        np.array -- 形状 [num, Block_rows, Block_cols] 的图像块数组
    """
    #* 参数初始化
    numImages, image_rows, image_cols = Images.shape
    block_rows, block_cols = ImageBlockSize
    half_block_rows, half_block_cols = block_rows // 2, block_cols // 2
    result = np.zeros([numSample, block_rows, block_cols])

    #* 随机生成选取的图片序号
    num_CaptureImage = [np.random.randint(numImages) for i in range(numSample)]

    for index, NumberOfBlocks in enumerate(num_CaptureImage):
        #* 获取第n个图片
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
        #* 截取图像
        Image_Block = Selected_Image[Selected_y - half_block_rows : Selected_y + half_block_rows + 1, Selected_x - half_block_cols: Selected_x + half_block_cols + 1]
        #* 存入数组
        result[index] = Image_Block
        #* 显示截取的图像
        # Load_MNIST.DisplayMNIST(Image_Block)
        if isLog > 0 and (index + 1) % isLog == 0:
            message = "Now Capturing Image Blocks... " + str(index + 1) + '/'+ str(numSample)
            PrintLog(message)

    PrintLog("Image Blocks Capture Done!")
    if isSavingData:
        #! 保存成文件
        np.save(isSavingData, result)

    return result

def LoadSamplingImageBlocks(filename):
    """读取保存的图像块数据
    
    Arguments:
        filename {string} -- 文件名， '*.npy'
    
    Returns:
        np.array -- 形状 [num, Block_rows, Block_cols] 的图像块数组
    """
    result = np.load(filename)
    return result

def GaborSingleImageBlock(Gabor_Filter, ImageBlock, method='same', whiten=True):
    """对单张图像进行滤波
    
    Arguments:
        Gabor_Filter {class} -- 吕强
        ImageBlock {np.array} -- 图像
    
    Keyword Arguments:
        whiten {bool} -- 是否白化 (default: {True})
        method {str} -- 卷积方法 (default: {True})
    
    Returns:
        np.array -- 滤波后的图像
    """
    result = Gabor_Filter.ConvImage(ImageBlock, method)
    return result

def GaborAllImageBlocks(Gabor_Filter, ImageBlocks, method='same', whiten=True, isSavingData=None, isLog=0):
    """对所有的图像块进行Gabor变换
    
    Arguments:
        Gabor_Filter {class} -- Gabor类
        ImageBlocks {np.array} -- 图像块
    
    Keyword Arguments:
        whiten {bool} -- 是否白化 (default: {True})
        isSavingData {str} -- 是否保存数据 (default: {None})
        isLog {int} -- 是否需要打log {default:{0}}
    
    Returns:
        np.array[numBlocks, numFilters, rows, cols] -- 返回值
    """
    numImages, rows, cols = ImageBlocks.shape
    numFilters = Gabor_Filter.GaborSize
    
    result = np.zeros([numImages, numFilters, rows, cols])
    for index, img in enumerate(ImageBlocks):
        result[index, :, :, :] = GaborSingleImageBlock(Gabor_Filter, img, method, whiten)
        if isLog > 0 and (index + 1) % isLog == 0:
            message = "Now Gaboring Image Blocks... " + str(index + 1) + '/'+ str(numImages)
            PrintLog(message)

    PrintLog("Image Blocks Gabor Done!")

    if isSavingData:
        #! 保存成文件
        np.save(isSavingData, result)
    
    return result

def LoadGaborImageBlocks(filename):
    """读取保存的Gabor图像块数据
    
    Arguments:
        filename {string} -- 文件名， '*.npy'
    
    Returns:
        np.array -- 形状 [numBlocks, numFilters, Block_rows, Block_cols] 的图像块数组
    """
    result = np.load(filename)
    return result


if __name__ == "__main__":
    Train_X, Train_Y, Test_X, Test_Y = Load_MNIST.Preprocess_MNIST_Data("./Datasets/MNIST_Data", True, True)
    savefile = 'ImageBlock.npy'
    ImageBlockSize = (11, 11)
    numSamples = 40
    
    #* 自动采样图像块
    Image_blocks = SamplingImageBlocks(Train_X, ImageBlockSize, numSamples, isSavingData=savefile, isLog=5)

    #* 读取已保存的图像块
    # res = LoadSamplingImageBlocks(savefile)

    ksize = (3, 3)
    Lambda = [1, 2, 3, 4]
    numTheta = 8
    Theta = [np.pi / numTheta * i for i in range(numTheta)]
    Beta = [0.5]
    Gamma = [1, 2]
    Gabor_Filter = GaborFilter.getGaborFilterClass(ksize, Theta, Lambda, Gamma, Beta, 'b')

    GaborAllImageBlocks(Gabor_Filter, Image_blocks, isLog=5)
