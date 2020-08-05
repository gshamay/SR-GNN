#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 5:40
# @Author : {ZM7}
# @updated : Gil Shamay and Alex Danieli
# @File : main.py
# @Software: PyCharm

from __future__ import division
import numpy as np
from tensorflow_code.model import *
from datasets.preprocessMethod import *
from tensorflow_code.utils import build_graph, Data, split_validation
import pickle
import argparse
import datetime
from printDebug import *
import time
import random

random.seed(1)
np.random.seed(1)

def trainAndTest(opt):
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

    printDebug("opt.dataset[" + opt.dataset + "]")
    item_Ctr = train_data[2]  # if you fail here - make sure you re prepare your DB [ this was added later]
    items_ctr_beforeAddition = train_data[3]
    n_node = item_Ctr

    listx = list(train_data)
    listx.pop(3)
    listx.pop(2)
    train_data = tuple(listx)

    trainSize = len(train_data[0])
    testSize = len(test_data[0])
    printDebug(
        "trainSize[" + str(trainSize)
        + "]testSize[" + str(testSize)
        + "]item_Ctr[" + str(item_Ctr)
        + "]items_ctr_beforeAddition[" + str(items_ctr_beforeAddition)
        + "]")

    train_data = Data(train_data, sub_graph=True, method=opt.method, shuffle=True)
    test_data = Data(test_data, sub_graph=True, method=opt.method, shuffle=False)
    model = GGNN(hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,
                 lr=opt.lr, l2=opt.l2, step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize,
                 lr_dc=opt.lr_dc,
                 nonhybrid=opt.nonhybrid)
    printDebug(str(opt))
    best_result = [0, 0]
    best_epoch = [0, 0]

    Start = datetime.datetime.now()
    startDateString = Start.strftime("%Y-%m-%d-%H-%M-%S")
    fileName = 'Main_TF_' + opt.dataset \
               + '_' + startDateString \
               + '_Es_' + str(opt.epoch) \
               + '_running'
    printDebug('-- Starting @' + startDateString)

    for epoch in range(opt.epoch):
        printDebug('epoch: ' + str(epoch) + '===========================================')
        slices = train_data.generate_batch(model.batch_size)
        fetches = [model.opt, model.loss_train, model.global_step]
        printDebug('start training: ' + str(datetime.datetime.now()))
        loss_ = []

        actualEOSs = 0
        predictedEOSs = 0
        falsePositiveEOSs = 0
        falseNegativeEOSs = 0

        for i, j in zip(slices, np.arange(len(slices))):
            adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
            _, loss, _ = model.run(fetches, targets, item, adj_in, adj_out, alias, mask)
            loss_.append(loss)
        loss = np.mean(loss_)
        slices = test_data.generate_batch(model.batch_size)
        printDebug('start predicting: ' + str(datetime.datetime.now()))
        hit, mrr, test_loss_ = [], [], []

        for i, j in zip(slices, np.arange(len(slices))):
            adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)
            scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, adj_in, adj_out, alias,
                                          mask)
            test_loss_.append(test_loss)
            index = np.argsort(scores, 1)[:, -20:]
            for score, target in zip(index, targets):
                # change score and target to be :
                # min (items_ctr_beforeAddition - 1) for score
                # and min (items_ctr_beforeAddition) for target (we reduce 1 for the target to fit the ids
                # (IDs count from 0, items count from 1)

                if items_ctr_beforeAddition != item_Ctr:
                    # added aEOS Items
                    originalScore = score
                    score = [min(x, items_ctr_beforeAddition - 1) for x in score]
                    originalTarge = target
                    target = min(target, items_ctr_beforeAddition)
                    targetIsAEOS = False
                    predictedAEOS = False
                    if target != originalTarge:
                        # can happen only if we are in EvalEOS_True!
                        _dummy = True
                    if len(set(score) & set(originalScore)) != set(originalScore):
                        # can happen only if we are in EOS > 0!
                        _dummy = True  # dummy value for debug

                    if target == items_ctr_beforeAddition:
                        targetIsAEOS = True
                        actualEOSs = actualEOSs + 1

                    if (items_ctr_beforeAddition - 1) in score:
                        predictedAEOS = True

                    if predictedAEOS and targetIsAEOS:
                        # we managed to predict aEOS correctly
                        predictedEOSs = predictedEOSs + 1
                    elif predictedAEOS:
                        # we predict aEOS - although it isn't
                        falsePositiveEOSs = falsePositiveEOSs + 1
                    elif targetIsAEOS:
                        # we predict Not aEOS - although it is
                        falseNegativeEOSs = falseNegativeEOSs + 1
                    else:
                        _nothing_special = True

                hit.append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (20 - np.where(score == target - 1)[0][0]))

        hit = np.mean(hit) * 100  # hit is calculating the P@20
        mrr = np.mean(mrr) * 100  # mrr is calculating the mrr@20
        test_loss = np.mean(test_loss_)
        if hit >= best_result[0]:  # best P@20 is stored in best_result[0] and it's best epoch at  best_epoch[0]
            best_result[0] = hit
            best_epoch[0] = epoch
        if mrr >= best_result[1]:  # best mrr@20 is stored in best_result[1] and it's best epoch at  best_epoch[1]
            best_result[1] = mrr
            best_epoch[1] = epoch

        printDebug('train_loss:\t%.4f\ttest_loss:\t%4f\tPrecision@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        printDebug('actualEOSs:\t%d\tpredictedEOSs:\t%d\tfalsePositiveEOSs:\t%d\tfalseNegativeEOSs\t%d' % (
            actualEOSs, predictedEOSs, falsePositiveEOSs, falseNegativeEOSs))

        printToFile(fileName)

    End = datetime.datetime.now()
    endDateString = End.strftime("%Y-%m-%d-%H-%M-%S")
    printDebug('Done @' + endDateString + " took " + str(End - Start))
    printToFile(fileName)
    finalFileName = 'Main_TF_' + opt.dataset \
                    + '_train_loss_%.4f_' % loss \
                    + '_test_loss_%4f_' % test_loss \
                    + '_Precision20_%.4f_' % best_result[0] \
                    + '_MRR20_%.4f_' % best_result[1] \
                    + '_Epoch_%d_' % best_epoch[0] \
                    + '_bestEpoch_%d' % best_epoch[1] \
                    + '_Es_' + str(opt.epoch) \
                    + '_' + str(startDateString)

    time.sleep(1)
    renameToFinalLog(fileName, finalFileName)

