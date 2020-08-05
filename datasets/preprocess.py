#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018
@author: Tangrizzly
@updated : Gil Shamay and Alex Danieli
"""
from datasets.preprocessMethod import *
import argparse

random.seed(1)

# Before running you must upadte the headers in youchoose clicks:
#   session_id,timestamp,item_id,ccategory (they are missing in the original data)
# the only used info from the data is the session and item IDs and the time/date
#
# This code generate the data to be used by the main.py
# it read the dataset and generate the train and test data files
# according to the specification given in the input parameter
# the input is the base dataset file and the parameters as described below
# the output of this code is one or more directories that will be named by the databse and it's different setting
# this directory will be the input of teh main.py
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
parser.add_argument('--minItemUsage', default='5', help='min item usage to be added to the graph, default is 5')
parser.add_argument('--minSeqLen', default='2', help='min seq length to be added to the graph. default is 2')

parser.add_argument('--EOS', default='0',
                    help='the rate of the EOS insertion, '
                         + '0 adds nothing '
                         + '1 add EOS for every real end, '
                         + 'default is 0. '
                         + 'if this value is set the aEOS are added to the TRAIN data')

parser.add_argument('--EOSNum', default='0',
                    help='the actual number of aEOSs that will be added, '
                         + '0 adds nothing ; any float is accepted [0,1]'
                         + 'this value overide the EOS ; '
                         + 'default is 0. '
                         + 'if this value is set the aEOS are added to the TRAIN data')

parser.add_argument('--EvalEOS', default='false',
                    help='(true) if evaluation should be done on all items, including last items'
                         + 'or (false) if should be done on all, except the last item in the seq, '
                         + 'as in the original code; default is false. '
                         + 'if this value is set the aEOS are added to the TEST data')

parser.add_argument('--EvalEOSTestOnly', default='false',
                    help='(true) if ONLY THE EVALUATION should be done on all items, including last itemd'
                         + 'or (false) if should be done as defined by the above parameters '
                         + 'This is to check the original code running on ALL data. '
                         + 'if this value is set the aEOS are added ONLY to the TEST data')
opt = parser.parse_args()

preprocess(opt)
