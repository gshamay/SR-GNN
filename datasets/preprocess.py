#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

# todo: [GS] added heades in youchoose clicks - session_id,timestamp,item_id,ccategory
# todo: [GS] add here end of session nodes EOS
# todo: should we look for the eos as a standard node ? change the validation to include last session, w/o the eos  ?
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset == 'yoochoose':
    dataset = 'yoochoose-clicks.dat'

print("-- Starting @ %ss" % datetime.datetime.now())
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

    print("create sess_clicks - session id --> Click and times ; and sess_date session id --> session date")
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:
            # sesssion switch detected
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date  # set the date of the session when there is a session switch
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        else:
            item = data['item_id'], int(data['timeframe'])
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
        # todo: [GS] yoochoose is sorted ?
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        print("modify sess_clicks - session id --> Click sorted by their times (without the time info)")
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]

    # todo: [GS] it is needed run only at the last session - bcz the last session date is not set (bcz there is no 'session switch')
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

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

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))  # todo: [GS] what is this used for ?
length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:  # todo: [GS] keep only ids that were clicked 5 times or more
        # todo: [GS] filter out sessions that has less then 2 clics that appeard less then 5 times (session become len 1)
        del sess_clicks[s]
        del sess_date[s]
        filteredOutSeq += 1
    else:
        sess_clicks[s] = filseq

print("Sequences before filtering out rare items and short sequences [" + str(length) + "]"
      + "after[" + str(len(sess_clicks)) + "]"
      + "filteredOutSeq[" + str(filteredOutSeq) + "]"
      )

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test # take the last 1 day from yoochosse or teh last 7 days of other dbs, and use them as the test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400 60*60*24
else:
    splitdate = maxdate - 86400 * 7

print('Splitting train/test date', splitdate)  # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
print("train sessions[" + str(len(tra_sess)) + "]")  # 186670    # 7966257
print("test  sessions[" + str(len(tes_sess)) + "]")  # 15979     # 15324
print(str(tra_sess[:3]) + "the first 3 sessions that will be used for train - out of [" + str(len(tra_sess)) + "]")
print(str(tes_sess[:3]) + "the first 3 sessions that will be used for test - out of [" + str(len(tes_sess)) + "]")
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}


# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
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
                item_dict[i] = item_ctr  # add the new item id to the dic (origItemID--> new ID)
                item_ctr += 1  # prepare the next new item id
        if len(outseq) < 2:  # Doesn't occur
            continue  # doesn't suppose to get here - bcz we filtered out all short sessions before
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print("number of items[" + str(item_ctr) + "]")  # 43098, 37484
    return train_ids, train_dates, train_seqs


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
            #     print("item not in the train dictionary-->ignored [" + i + "]")
        if len(outseq) < 2:
            continue  # we can get here, bcz we filtered out items that are not in the training
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


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
            ids += [id]
    return out_seqs, out_dates, labs, ids


trainSeqNumBefore = (len(tra_seqs))
testSeqNumBefore = (len(tes_seqs))
# generate new Seq based on all sub seq of the given seq --> We can HERE our improvement
tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print("Train size : before addiongSubSeq[" + str(trainSeqNumBefore) + "]after[" + str(len(tr_seqs)) + "]")
print("Test  size : before addiongSubSeq[" + str(testSeqNumBefore) + "]after[" + str(len(te_seqs)) + "]")
print("Examples:")
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])  # example for train
print(te_seqs[:3], te_dates[:3], te_labs[:3])  # example for test
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

print('avg Train: ', allTarin / (len(tra_seqs) * 1.0))
print('avg Test: ', allTest / (len(tes_seqs) * 1.0))
print('avg length - all: ', all / (len(tra_seqs) + len(tes_seqs) * 1.0))

if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    # use only the last 1/4 of yoochoose @ yoochoose1_4 and 1/64 in yoochoose1_64
    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')
