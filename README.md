# SR-GNN 

## Paper data and code

This is the code was originally developed for the AAAI 2019 Paper: [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855). 

It was updated by:

Gil Shamay(gshamay@gmail.com) and Alex Danieli (alex.daniels85@gmail.com)

RecSys course project, Spring 2020, 

Department of Information Systems Engineering

Ben-Gurion University, Beer-Sheva, Israel

The original code is located @ https://github.com/CRIPAC-DIG/SR-GNN

The code was updated in order to extended it's abilities as described in a paper that will be updated here soon 

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `datasets/`:

- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup>

There is a small dataset `sample` included in the folder `datasets/`, which can be used to test the correctness of the code.

SR-GNN authors also wrote a [blog](https://sxkdz.github.io/research/SR-GNN) explaining the paper


## Usage

You need to run the file  `datasets/preprocess.py` first to preprocess the data.

For example: `cd datasets; python preprocess.py --dataset=sample`

```bash
usage: preprocess.py [-h] [--dataset DATASET] [--minItemUsage MINITEMUSAGE]
                     [--minSeqLen MINSEQLEN] [--EOS EOS] [--EOSNum EOSNUM]
                     [--EvalEOS EVALEOS] [--EvalEOSTestOnly EVALEOSTESTONLY]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name: diginetica/yoochoose/sample
  --minItemUsage MINITEMUSAGE
                        min item usage to be added to the graph, default is 5
  --minSeqLen MINSEQLEN
                        min seq length to be added to the graph. default is 2
  --EOS EOS             the rate of the EOS insertion, 0 adds nothing 1 add
                        EOS for every real end, default is 0. if this value is
                        set the aEOS are added to the TRAIN data
  --EOSNum EOSNUM       the actual number of aEOSs that will be added, 0 adds
                        nothing ; any float is accepted [0,1]this value
                        overide the EOS ; default is 0. if this value is set
                        the aEOS are added to the TRAIN data
  --EvalEOS EVALEOS     (true) if evaluation should be done on all items,
                        including last itemsor (false) if should be done on
                        all, except the last item in the seq, as in the
                        original code; default is false. if this value is set
                        the aEOS are added to the TEST data
  --EvalEOSTestOnly EVALEOSTESTONLY
                        (true) if ONLY THE EVALUATION should be done on all
                        items, including last itemdor (false) if should be
                        done as defined by the above parameters This is to
                        check the original code running on ALL data. if this
                        value is set the aEOS are added ONLY to the TEST data



  
  
```
the preprocess generate a directory with data files, in .\datasets\ directory

the directory name will contain some of the optional parameters

for example: datasets\digineticaEOS_0.0_EOSNum_1_EvalEOS_False_bEvalEOSTestOnly_False



Then you can run the file `tensorflow_code/main.py` to train the model.

use the generated directory name as the name of the dataset used as the input for the main

For example: `cd tensorflow_code; python main.py --dataset = digineticaEOS_0.0_EOSNum_1_EvalEOS_False_bEvalEOSTestOnly_False`

You can add the suffix `--nonhybrid` to use the global preference of a session graph to recommend instead of the hybrid preference.

You can also change other parameters according to the usage:

```bash
usage: main.py [-h] [--dataset DATASET] [--batchSize BATCHSIZE]
               [--hiddenSize HIDDENSIZE] [--epoch EPOCH] [--lr LR]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2]
               [--step STEP] [--patience PATIENCE] [--nonhybrid]
               [--validation] [--valid_portion VALID_PORTION]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name:
                        diginetica/yoochoose1_4/yoochoose1_64/sample
  --batchSize BATCHSIZE
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of epochs after which the learning rate
                        decay
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  --nonhybrid           only use the global preference to predict
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set
```

## Requirements

- Python 3


## Citation

original paper:

```
@inproceedings{Wu:2019ke,
title = {{Session-based Recommendation with Graph Neural Networks}},
author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
year = 2019,
booktitle = {Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence},
location = {Honolulu, HI, USA},
month = jul,
volume = 33,
number = 1,
series = {AAAI '19},
pages = {346--353},
url = {https://aaai.org/ojs/index.php/AAAI/article/view/3804},
doi = {10.1609/aaai.v33i01.3301346},
editor = {Pascal Van Hentenryck and Zhi-Hua Zhou},
}
```

