'''
@Author: Cabrite
@Date: 2020-07-04 22:31:39
@LastEditors: Cabrite
@LastEditTime: 2020-07-11 10:31:03
@Description: 日志操作函数
'''

import datetime
import shutil
import sys
import os

def getTimeInfo():
    """获取时间信息

    Returns:
        str: 时间
    """
    return "[" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + "] "

class TFprint():
    #- 静态变量
    logger_file_path = ''
    isWritingFile = False
    logfile = None
    Dir_Path = './Console Logs'
    auto_line_break=True

    #- 静态函数
    @staticmethod
    def CreateLogFile(File_Name='', File_Append="Log", Dir_Path='./Console Logs', auto_line_break=True):
        """创建日志文件
        """
        if TFprint.logger_file_path:
            PrintLog("One Logging file processing. Unable to create until it's done!")
            return

        TFprint.Dir_Path = Dir_Path
        TFprint.auto_line_break = auto_line_break
        if not os.path.exists(Dir_Path):
            os.mkdir(Dir_Path)
        if not File_Name:
            File_Name = "[" + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + "] " + File_Append + ".txt"
        TFprint.logger_file_path = os.path.join(Dir_Path, File_Name)
        TFprint.__open_log_file()

    @staticmethod
    def RemoveCurrentLogFile():
        """删除当前日志文件
        """
        if TFprint.isWritingFile:
            TFprint.__close_log_file()
        if os.path.exists(TFprint.logger_file_path):
            os.remove(TFprint.logger_file_path)
            TFprint.logger_file_path = ''
            TFprint.isWritingFile = False
            TFprint.logfile = None
            TFprint.Dir_Path = './Console Logs'
            TFprint.auto_line_break=True
        else:
            PrintLog("Warning! No such directory existing.")

    @staticmethod
    def RemoveLogFloder(dir_path = ''):
        """删除日志文件夹

        Args:
            dir_path (str, optional): 指定文件夹. Defaults to ''.
        """
        if TFprint.isWritingFile:
            TFprint.__close_log_file()
        
        delete_path = TFprint.Dir_Path if not dir_path else dir_path

        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)
            TFprint.logger_file_path = ''
            TFprint.isWritingFile = False
            TFprint.logfile = None
            TFprint.Dir_Path = './Console Logs'
            TFprint.auto_line_break=True
        else:
            PrintLog("Warning! No such directory existing.")
    
    @staticmethod
    def Fprint(message, with_time=True):
        """向日志文件中写入日志

        Args:
            message (str): 日志内容
        """
        if not TFprint.logger_file_path:
            TFprint.CreateLogFile()
        if TFprint.auto_line_break:
            message = message + '\n'
        if with_time:
            message = getTimeInfo() + message
        TFprint.logfile.write(message)
    
    @staticmethod
    def TFprint(message, diff_logTime=None, with_time=True):
        """文件、控制台双输出

        Args:
            message (str): 日志内容
        """
        TFprint.Fprint(message, with_time)
        return PrintLog(message, diff_logTime)

    #- 开关Log文件
    @staticmethod
    def __open_log_file():
        """打开日志文件
        """
        TFprint.isWritingFile = True
        TFprint.logfile = open(TFprint.logger_file_path, "a+")

    @staticmethod
    def __close_log_file():
        """关闭日志文件
        """
        TFprint.isWritingFile = False
        TFprint.logfile.close()
    
    @staticmethod
    def FinishLogging():
        """结束日志
        """
        TFprint.__close_log_file()

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

def ProcessingBar(Counting, Total, graph = '█', CompleteLog = 'Success!', isClear = False, with_time = True):
    """进度条，一旦使用，Terminal窗口不能调整大小，且不能太小

    Args:
        Counting (int): 计数
        Total (int): 总数
        graph (str, optional): 进度条形状. Defaults to '█'.
        CompleteLog (str, optional): 结束日志标志. Defaults to 'Success!'.
        isClear (bool, optional): 是否进度结束时清楚进度条. Defaults to False.
        with_time (bool, optional): 是否带时间. Defaults to True.
    """
    terminal_width = os.get_terminal_size().columns
    if with_time:
        terminal_width = terminal_width - 30

    processingbar_width = int(terminal_width / 8 * 5)
    
    complete_ratio = float(Counting) / Total
    complete_part = int(processingbar_width * complete_ratio)
    uncomplete_part = processingbar_width - complete_part

    msg = getTimeInfo() if with_time else ""
    msg = msg + '|' + graph * complete_part + "_" * uncomplete_part + '| {:.2f}%\r'.format(float(Counting) / Total * 100)
    print(msg, end = "")

    if Counting == Total:
        #* 如果清空，则将terminal宽的字符全部清除
        if isClear:
            print(' ' * (os.get_terminal_size().columns - 1) + '\r', end = "")
        #* 如果输入结束日志，则显示结束结果
        elif CompleteLog:
            print("\r\n" + getTimeInfo() + CompleteLog)
        #* 如果没有结束日志，则直接换行
        else:
            print()


if __name__ == "__main__":
    #- 时间日志测试
    msg = PrintLog("Testing Console Time Start")
    PrintLog("Testing Console Time End", msg)

    #- 进度条测试
    import time
    for i in range(200):
        ProcessingBar(i + 1, 200, '*', CompleteLog="AC")
        time.sleep(0.005)

    #- 文本日志测试
    TFprint.CreateLogFile()
    TFprint.Fprint("123")
    TFprint.Fprint("456")
    TFprint.Fprint("789")
    TFprint.TFprint("987")
    TFprint.TFprint("654")
    TFprint.TFprint("321")
    
    # tfp.FinishLogging()

    #* 删除当前日志目录
    TFprint.RemoveLogFloder("./Console Logs1")
    # TFprint.RemoveLogFloder()

    time.sleep(2)
    #* 删除当前日志文件
    TFprint.RemoveCurrentLogFile()




