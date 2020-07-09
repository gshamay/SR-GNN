# Alex Danieli
# Gil Shamay
import datetime

stringToPrintToFile = ""


def printDebug(str):
    global stringToPrintToFile
    dateString = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    str = dateString + " " + str
    print(str)
    stringToPrintToFile = stringToPrintToFile + str + "\r"


def printToFile(fileName):
    global stringToPrintToFile
    file1 = open(fileName, "a")
    file1.write("\r************************\r")
    file1.write(stringToPrintToFile)
    stringToPrintToFile = ""
    file1.close()
