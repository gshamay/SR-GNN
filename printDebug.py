# Alex Danieli
# Gil Shamay
import datetime
import os
import matplotlib.pyplot as plt

stringToPrintToFile = ""


def printDebug(str):
    global stringToPrintToFile
    dateString = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    str = dateString + " " + str
    print(str)
    stringToPrintToFile = stringToPrintToFile + str + "\r"


destinationFolder = "./../testResults/"

_FileName_ = ""


def setFileName(fileName):
    global _FileName_
    _FileName_ = fileName


def printToFile(fileName=""):
    if (fileName == ""):
        fileName = _FileName_
    fileName = fileNameToFullPath(fileName)
    global stringToPrintToFile
    file1 = open(fileName, "a")
    file1.write("\r************************\r")
    file1.write(stringToPrintToFile)
    stringToPrintToFile = ""
    file1.close()


def fileNameToFullPath(fileName, ext=".log"):
    fileName = destinationFolder + fileName + ext
    return fileName


def renameToFinalLog(src, trgt):
    os.rename(fileNameToFullPath(src), fileNameToFullPath(trgt))


def plotToFile(fileName):
    plt.savefig(fileNameToFullPath(fileName, '.png'))
