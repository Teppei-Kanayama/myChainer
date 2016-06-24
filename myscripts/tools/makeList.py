# -*- coding:utf-8 -*-

import os

path = "/data/ishimochi2/dataset/ILSVRC2012_img_train_for_caffe/"
dirlist = os.listdir(path)
class_counter = 0

f = open("../train_dataList.txt", "w")
for directory in dirlist:
    tmppath = path + directory + '/'
    tmpdirlist = os.listdir(tmppath)
    #classname = directory.split('n')[1]
    for filename in tmpdirlist:
        #tmp = tmppath + filename + " " + classname + "\n"
        tmp = tmppath + filename + " " + str(class_counter) + "\n"
        #print tmp
        f.write(tmp)
    class_counter += 1
f.close()

    
