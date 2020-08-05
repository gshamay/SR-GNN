#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 5:40
# @Author : {ZM7}
# @updated : Gil Shamay and Alex Danieli
# @File : main.py
# @Software: PyCharm

from __future__ import division

from datasets.preprocessMethod import *
from tensorflow_code.mainMethod import *

import argparse
import datetime
from printDebug import *
import time
import random

random.seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample',
                    help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample/sampleEOS_0.2_EvalEOS_False')
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--runner', type=int, default=0, help='run the loop experiment')
opt = parser.parse_args()

if opt.runner == 1:
    class genericClass():
        a = 1


    opt_preprocess = genericClass()
    opt_preprocess.dataset = 'sample'
    opt_preprocess.minItemUsage = '5'
    opt_preprocess.minSeqLen = '2'
    opt_preprocess.EOS = '0'
    opt_preprocess.EOSNum = '1'
    opt_preprocess.EvalEOS = 'false'
    opt_preprocess.EvalEOSTestOnly = 'false'

    preprocess(opt_preprocess)

    opt_trainAndTest = genericClass()
    opt_trainAndTest.dataset = opt_preprocess.dataset \
                               + 'EOS_0.0' \
                               + '_EOSNum_' + opt_preprocess.EOSNum \
                               + '_EvalEOS_' + opt_preprocess.EvalEOS \
                               + '_bEvalEOSTestOnly_' + opt_preprocess.EvalEOSTestOnly

    opt_trainAndTest.method = opt.method
    opt_trainAndTest.validation = opt.validation
    opt_trainAndTest.epoch = opt.epoch
    opt_trainAndTest.batchSize = opt.batchSize
    opt_trainAndTest.hiddenSize = opt.hiddenSize
    opt_trainAndTest.l2 = opt.l2
    opt_trainAndTest.lr = opt.lr
    opt_trainAndTest.step = opt.step
    opt_trainAndTest.nonhybrid = opt.nonhybrid
    opt_trainAndTest.lr_dc = opt.lr_dc
    opt_trainAndTest.lr_dc_step = opt.lr_dc_step

    trainAndTest(opt_trainAndTest)
else:
    trainAndTest(opt)
