#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018
@author: Tangrizzly
@updated : Gil Shamay and Alex Danieli
"""
import argparse
import csv
import operator
import pickle
import random
import time

from printDebug import *

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

def preprocess(opt):
    printDebug("opt=" + str(opt))

    session_id = 'session_id'  # those fields has a bit different names between the different DBs
    item_id = 'item_id'
    dataset = 'sample_train-item-views.csv'
    if opt.dataset == 'diginetica':
        dataset = 'train-item-views.csv'
        session_id = 'sessionId'
        item_id = 'itemId'
    elif opt.dataset == 'yoochoose':
        dataset = 'yoochoose-clicks.dat'

    minItemUsage = int(opt.minItemUsage)
    if minItemUsage <= 0:
        printDebug("bad minItemUsage value [" + opt.minItemUsage + "] setting to 5")
        minItemUsage = 5
    printDebug("minItemUsage[" + str(minItemUsage) + "]")

    minSeqLen = int(opt.minSeqLen)
    if minSeqLen < 0:
        printDebug("bad minSeqLen value [" + opt.minSeqLen + "] setting to 2")
        minSeqLen = 2
    printDebug("minSeqLen[" + str(minSeqLen) + "]")

    fEOS = 0.0
    if opt.EOS == '0':
        fEOS = 0.0
    else:
        fEOS = float(opt.EOS)

    printDebug("fEOS[" + str(fEOS) + "]")

    iEOSNum = 0
    if opt.EOSNum == '0':
        iEOSNum = 0
    else:
        iEOSNum = int(opt.EOSNum)

    printDebug("iEOSNum[" + str(iEOSNum) + "]")

    bEvalEOS = False
    if opt.EvalEOS == 'false':
        bEvalEOS = False
    else:
        bEvalEOS = True

    printDebug("bEvalEOS[" + str(bEvalEOS) + "]")

    bEvalEOSTestOnly = False
    if opt.EvalEOSTestOnly == 'false':
        bEvalEOSTestOnly = False
    else:
        bEvalEOSTestOnly = True

    printDebug("bEvalEOSTestOnly[" + str(bEvalEOSTestOnly) + "]")
    ##############################################
    Start = datetime.datetime.now()
    dateBeginString = Start.strftime("%Y-%m-%d-%H%M%S")

    fileName = "preprocess_" + opt.dataset \
               + "_EOS_" + str(fEOS) \
               + "_EOSNum_" + str(iEOSNum) \
               + "_EvalEOS_" + str(bEvalEOS) \
               + "_bEvalEOSTestOnly_" + str(bEvalEOSTestOnly) \
               + "_minItemUsage_" + str(minItemUsage) \
               + "_minSeqLen_" + str(minSeqLen) \
               + "_" + dateBeginString

    setFileName(fileName)
    printDebug('-- Starting @' + dateBeginString)

    with open(dataset, "r") as f:
        if opt.dataset == 'yoochoose':
            reader = csv.DictReader(f, delimiter=',')
        else:
            reader = csv.DictReader(f, delimiter=';')
        sess_clicks = {}  # session id --> Sortred clicked items
        sess_date = {}  # session id --> click times
        ctr = 0
        curid = -1
        curdate = None

        printDebug("create sess_clicks - session id --> Click and times ; and sess_date session id --> session date")
        for data in reader:
            sessid = data[session_id]
            if curdate and not curid == sessid:
                # session switch detected
                date = ''
                if opt.dataset == 'yoochoose':
                    date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
                else:
                    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                sess_date[curid] = date  # set the date of the session when there is a session switch
            curid = sessid
            if opt.dataset == 'yoochoose':
                item = data[item_id]
            else:
                item = data[item_id], int(data['timeframe'])
            curdate = ''
            if opt.dataset == 'yoochoose':
                curdate = data['timestamp']
            else:
                curdate = data['eventdate']

            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1

        date = ''
        if opt.dataset == 'yoochoose':
            date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
        else:
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            printDebug("modify sess_clicks - session id --> Click sorted by their times (without the time info)")
            for i in list(sess_clicks):
                sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
                sess_clicks[i] = [c[0] for c in sorted_clicks]

        sess_date[curid] = date
    printDebug("-- Reading data @ %ss" % datetime.datetime.now())

    # Filter out length 1 sessions
    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            del sess_clicks[s]
            del sess_date[s]

    # Count number of times each item appears
    iid_counts = {}  # item --> Count num of times that the item was used in all seq
    filteredOutSeq = 0
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1

    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))  # var not in use
    length = len(sess_clicks)
    for s in list(sess_clicks):
        curseq = sess_clicks[s]
        filseq = list(filter(lambda i: iid_counts[i] >= minItemUsage,
                             curseq))  # filter out items that appeard less then minItemUsage(default is 5) times
        if len(filseq) < minSeqLen:
            # keep only ids that were clicked minSeqLen (default is 2) times or more
            # filter out sessions that has less then minSeqLen clicks
            del sess_clicks[s]
            del sess_date[s]
            filteredOutSeq += 1
        else:
            sess_clicks[s] = filseq

    printDebug("filtered out rare items and short sequences "
               + "before[" + str(length) + "]"
               + "after[" + str(len(sess_clicks)) + "]"
               + "filteredOutSeq[" + str(filteredOutSeq) + "]"
               )

    ##############################################
    printDebug("Split out test set based on dates")
    dates = list(sess_date.items())
    maxdate = dates[0][1]

    for _, date in dates:
        if maxdate < date:
            maxdate = date

    # 7 days for test # take the last 1 day from yoochosse or the last 7 days of other dbs, and use them as the test
    splitdate = 0
    if opt.dataset == 'yoochoose':
        splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400 60sec*60Min*24hours
    else:
        splitdate = maxdate - 86400 * 7

    printDebug('Splitting train/test date' + str(splitdate))  # Yoochoose: ('Split date', 1411930799.0)
    tra_sess = filter(lambda x: x[1] < splitdate, dates)
    tes_sess = filter(lambda x: x[1] > splitdate, dates)

    # Sort sessions by date
    tra_sess = sorted(tra_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
    tes_sess = sorted(tes_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
    printDebug("train sessions[" + str(len(tra_sess)) + "]")  # 186670    # 7966257
    printDebug("test  sessions[" + str(len(tes_sess)) + "]")  # 15979     # 15324
    printDebug(str(tra_sess[:3]) + "the first 3 sessions that will be used for train - out of [" + str(len(tra_sess)) + "]")
    printDebug(str(tes_sess[:3]) + "the first 3 sessions that will be used for test - out of [" + str(len(tes_sess)) + "]")
    printDebug("-- Splitting train set and test set @ %ss" % datetime.datetime.now())  # the Split us done by the session id

    ##############################################
    # Convert training sessions to sequences and renumber items to start from 1
    # Choosing item count >=5 gives approximately the same number of items as reported in paper
    item_dict = {}


    def obtian_tra():
        train_ids = []  # the sessions/seq ids
        train_seqs = []  # the seqs after the convert
        train_dates = []  # all sessions dates
        item_ctr = 1
        for s, date in tra_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    # item_dict contain the new id of the item - if this item was already uesed- re use this new item id
                    outseq += [item_dict[i]]
                else:
                    outseq += [item_ctr]
                    item_dict[i] = item_ctr  # add the new item id to the dic (origItemID --> new ID)
                    item_ctr += 1  # prepare the next new item id
            if len(outseq) < minSeqLen:  # Doesn't occur
                printDebug("Error ! should not get here")
                continue  # doesn't suppose to get here - bcz we filtered out all short sessions before
            train_ids += [s]
            train_dates += [date]
            train_seqs += [outseq]
        printDebug("number of items[" + str(item_ctr) + "]")  # 43098, 37484
        return train_ids, train_dates, train_seqs, item_ctr


    # Convert test sessions to sequences, ignoring items that do not appear in training set
    def obtian_tes():
        test_ids = []
        test_seqs = []
        test_dates = []
        for s, date in tes_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:  # if item is NOT in the dic of the items in the train --> it is ignored
                    outseq += [item_dict[i]]
                # else:
                #     printDebug("item not in the train dictionary-->ignored [" + i + "]")
            if len(outseq) < minSeqLen:
                continue  # we can get here, bcz we filtered out items that are not in the training
            test_ids += [s]
            test_dates += [date]
            test_seqs += [outseq]
        return test_ids, test_dates, test_seqs


    tra_ids, tra_dates, tra_seqs, items_ctr = obtian_tra()
    tes_ids, tes_dates, tes_seqs = obtian_tes()

    ##############################################
    # find how often the same EOS item is used ; there may some that are 'natural' EOS (like checkout page)
    # this must be done on the train only - we can't know in the test what is the EOS
    # however we should add the aEOSs (artificial EOSs) to the test
    # and this must be done before adding sub sequences

    iid_EOS_counts = {}  # item --> Count num of times that the item was used as EOS
    for curseq in tra_seqs:
        # count the number of times that item appear as the item in the seq
        lastItemInSession = curseq[len(curseq) - 1]
        if lastItemInSession in iid_EOS_counts:
            iid_EOS_counts[lastItemInSession] += 1
        else:
            iid_EOS_counts[lastItemInSession] = 1

    printDebug("EOS items in train[" + str(len(iid_EOS_counts)) + "]")

    iid_EOS_counts_sorted = sorted(iid_EOS_counts.items(), key=lambda x: x[1], reverse=True)
    EOS_counts = []
    for x in iid_EOS_counts_sorted:
        EOS_counts.append(x[1])

    numOfEOSToAdd = 0
    if fEOS > 0.0:
        numOfEOSToAdd = int(fEOS * len(iid_EOS_counts))
        if numOfEOSToAdd < 1:
            numOfEOSToAdd = 1
        if bEvalEOSTestOnly:
            numOfEOSToAdd = 1

    if iEOSNum > 0:
        numOfEOSToAdd = iEOSNum

    printDebug("Sequences in train [" + str(len(tra_seqs)) + "]"
               + "EOS items[" + str(len(iid_EOS_counts)) + "]"
               + "MaxEOSLinks[" + str(max(EOS_counts)) + "]"
               + "fEOS[" + str(fEOS) + "]"
               + "iEOSNum[" + str(iEOSNum) + "]"
               + "numOfEOSToAdd[" + str(numOfEOSToAdd) + "]aEOSs"
               )

    # add THE artificial EOS to the train - they can't be added as negative items (algorithm expect positive IDs)
    # --> use items_ctr as the ID generator
    items_ctr_beforeAddition = items_ctr
    addedEOSsOnTrain_IDs = {}  # keep the IDs of the aEOSs ; not all expected items may be added
    addedEOSsOnTrain = {}  # keep statistics about the added aESOs / train
    addedEOSsOnTest = {}  # keep statistics about the added aESOs / test

    # update train with aEOSs
    if not bEvalEOSTestOnly:
        if numOfEOSToAdd > 0:
            for curseq in tra_seqs:
                if numOfEOSToAdd == 1:
                    eosToAdd = -1
                else:
                    eosToAdd = -(random.randrange(1, numOfEOSToAdd))

                if eosToAdd in addedEOSsOnTrain:
                    addedEOSsOnTrain[eosToAdd] += 1
                else:
                    addedEOSsOnTrain[eosToAdd] = 1
                    addedEOSsOnTrain_IDs[eosToAdd] = items_ctr
                    items_ctr = items_ctr + 1

                if eosToAdd in addedEOSsOnTrain_IDs:
                    curseq.append(addedEOSsOnTrain_IDs[eosToAdd])
                else:
                    printDebug("ERROR! We must not get here (bad aEOS in train)")

    plt.hist(EOS_counts, bins=max(EOS_counts))
    plt.yscale('log')
    plotToFile(fileName + "_FullHistogram")

    # if we want to validate as in the original paper
    # (without taking in consideration prediction on the last items)
    # we should not add the aEOSs to the test
    # but of we want to check the results WITH the actual EOS items - we should add the aESOs
    # those are used only for evaluation... we add them before adding the sub sequences

    # update Test with aEOSs

    bAddedASingleAEOSForAll = False
    if bEvalEOS:
        if numOfEOSToAdd > 0:
            for curseq in tes_seqs:
                if numOfEOSToAdd == 1:
                    eosToAdd = -1
                else:
                    eosToAdd = -(random.randrange(1, numOfEOSToAdd))

                if eosToAdd in addedEOSsOnTrain_IDs:
                    curseq.append(addedEOSsOnTrain_IDs[eosToAdd])
                else:
                    if bEvalEOSTestOnly:
                        # we didn't add any aEOS on train - only in test - we will add a single one for all -
                        # it's not used for training at all
                        curseq.append(items_ctr)
                        bAddedASingleAEOSForAll = True
                        addedEOSsOnTest[eosToAdd] = 1
                    else:
                        printDebug("ERROR! We must not get here (bad aEOS in test)")
                if eosToAdd in addedEOSsOnTest:
                    addedEOSsOnTest[eosToAdd] += 1
                else:
                    addedEOSsOnTest[eosToAdd] = 1

    if bAddedASingleAEOSForAll:
        items_ctr = items_ctr + 1

    printDebug("items_ctr[" + str(items_ctr)
               + "]actual added aEOSs on train[" + str(len(addedEOSsOnTrain))
               + "]actual added aEOSs on Test[" + str(len(addedEOSsOnTest))
               + "]items_ctr_beforeAddition[" + str(items_ctr_beforeAddition)
               + "]")


    ##############################################
    # generate new Seq based on all sub seq of the given seq
    def process_seqs(iseqs, idates):
        out_seqs = []
        out_dates = []
        labs = []
        ids = []
        for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
                out_dates += [date]
                ids += [id]  # The same session ID can be added a few times - once for each sub seq
                #  But those are new Session IDS now
        return out_seqs, out_dates, labs, ids


    trainSeqNumBefore = (len(tra_seqs))
    testSeqNumBefore = (len(tes_seqs))
    tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
    te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
    tra = (tr_seqs, tr_labs, items_ctr, items_ctr_beforeAddition)
    tes = (te_seqs, te_labs)
    printDebug("Train size : before adding Sub Seq[" + str(trainSeqNumBefore) + "]after[" + str(len(tr_seqs)) + "]")
    printDebug("Test  size : before adding Sub Seq[" + str(testSeqNumBefore) + "]after[" + str(len(te_seqs)) + "]")

    printDebug("Examples:")
    printDebug(str(tr_seqs[:3]) + "," + str(tr_dates[:3]) + str(tr_labs[:3]))  # example for train
    printDebug(str(te_seqs[:3]) + "," + str(te_dates[:3]) + "," + str(te_labs[:3]))  # example for test

    all = 0
    allTarin = 0
    allTest = 0

    # Calculate avg seq lengths
    for seq in tra_seqs:
        all += len(seq)
        allTarin += len(seq)
    for seq in tes_seqs:
        all += len(seq)
        allTest += len(seq)

    printDebug('avg Train: ' + str(allTarin / (len(tra_seqs) * 1.0)))  # those values are BEFORE adding the sub Seqs
    printDebug('avg Test: ' + str(allTest / (len(tes_seqs) * 1.0)))
    printDebug('avg length - all: ' + str(all / (len(tra_seqs) + len(tes_seqs) * 1.0)))

    pathExt = ""
    # when we add EOS items, we saved them as a new DB info, according to the rate of the added items
    if numOfEOSToAdd > 0:
        pathExt = "EOS_" + str(fEOS) \
                  + "_EOSNum_" + str(iEOSNum) \
                  + "_EvalEOS_" + str(bEvalEOS) \
                  + "_bEvalEOSTestOnly_" + str(bEvalEOSTestOnly)

    if opt.dataset == 'diginetica':
        if not os.path.exists('diginetica' + pathExt):
            os.makedirs('diginetica' + pathExt)
        pickle.dump(tra, open('diginetica' + pathExt + '/train.txt', 'wb'))
        pickle.dump(tes, open('diginetica' + pathExt + '/test.txt', 'wb'))
        pickle.dump(tra_seqs, open('diginetica' + pathExt + '/all_train_seq.txt', 'wb'))  # not in use
    elif opt.dataset == 'yoochoose':
        if not os.path.exists('yoochoose1_4' + pathExt):
            os.makedirs('yoochoose1_4' + pathExt)
        if not os.path.exists('yoochoose1_64' + pathExt):
            os.makedirs('yoochoose1_64' + pathExt)
        pickle.dump(tes, open('yoochoose1_4' + pathExt + '/test.txt', 'wb'))
        pickle.dump(tes, open('yoochoose1_64' + pathExt + '/test.txt', 'wb'))

        # use only the last 1/4 of yoochoose @ yoochoose1_4 and 1/64 in yoochoose1_64
        split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
        printDebug("1/4  db - train seq " + str(split4))
        printDebug("1/64 db - train seq " + str(split64))
        tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:], items_ctr, items_ctr_beforeAddition), (
            tr_seqs[-split64:], tr_labs[-split64:], items_ctr, items_ctr_beforeAddition)
        seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

        pickle.dump(tra4, open('yoochoose1_4' + pathExt + '/train.txt', 'wb'))
        pickle.dump(seq4, open('yoochoose1_4' + pathExt + '/all_train_seq.txt', 'wb'))  # not in use

        pickle.dump(tra64, open('yoochoose1_64' + pathExt + '/train.txt', 'wb'))
        pickle.dump(seq64, open('yoochoose1_64' + pathExt + '/all_train_seq.txt', 'wb'))  # not in use

    else:
        if not os.path.exists('sample' + pathExt):
            os.makedirs('sample' + pathExt)
        pickle.dump(tra, open('sample' + pathExt + '/train.txt', 'wb'))
        pickle.dump(tes, open('sample' + pathExt + '/test.txt', 'wb'))
        pickle.dump(tra_seqs, open('sample' + pathExt + '/all_train_seq.txt', 'wb'))  # not in use
        # tra_seqs - var not in use (the original seq before the sub seq)

    End = datetime.datetime.now()
    dateEndString = End.strftime("%Y-%m-%d-%H-%M-%S")
    printDebug('Done @' + dateEndString + " took " + str(End - Start))
    printToFile()

    # vars not in use: tra_ids, tes_ids, seq4, seq64, tra_seqs, tr_dates, tr_ids te_dates,te_ids


preprocess(opt)