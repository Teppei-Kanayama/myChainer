# coding: utf-8

import os
from PIL import Image
import numpy as np

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.JPEG')]

def imresize(im, sz):
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize(sz))

def resize(infile_name, outfile_name, width, hight):
    Image.open(infile_name).resize((width, hight)).save(outfile_name)
    print "finish"

def histeq(im, nbr_bins=256):
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum() #累積分布関数
    cdf = 255 * cdf / cdf[-1] #正規化
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def compute_average(imlist):
    """画像列の平均を求める"""

    averageim = np.array(Image.open(imlist[0]), 'f') #最初の画像を開いて浮動小数点に変換
    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print imname + '...skiped'
    averageim /= len(imlist)

    return np.array(averageim, 'uint8')
