# Alex Danieli
# Gil Shamay
import datetime
import os

stringToPrintToFile = ""


def printDebug(str):
    global stringToPrintToFile
    dateString = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    str = dateString + " " + str
    print(str)
    stringToPrintToFile = stringToPrintToFile + str + "\r"


destinationFolder = "./../testResults/"


def printToFile(fileName):
    fileName = fileNameToFullPath(fileName)
    global stringToPrintToFile
    file1 = open(fileName, "a")
    file1.write("\r************************\r")
    file1.write(stringToPrintToFile)
    stringToPrintToFile = ""
    file1.close()


def fileNameToFullPath(fileName):
    fileName = destinationFolder + fileName + ".log"
    return fileName


def renameToFinalLog(src, trgt):
    os.rename(fileNameToFullPath(src), fileNameToFullPath(trgt))
