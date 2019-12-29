#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sklearn.metrics as metrics
import random
import numpy as np
random.seed(1)
np.random.seed(1)

def Coverage(decval_output, groudtruth, example_pivot=True):
    """
      Evaluate coverage in example pivot or label pivot mode.

      Arguments:
        decval_output:  decsion value matrix (example x label matrix).

        groudtruth: binary groudtruth matrix (example x label matrix).

        example_pivot:True if average precision is calculated in example-pivot, False
        if we calculate label-pivot average precision.

      Return:
        coverage.
    """
    # coverage = 1/m sum_i max(r_i(lambda)) with lambda \in (groudtruth_i == 1).
    if (not example_pivot):
        decval_output = decval_output.T  # output is label x example
        groudtruth = groudtruth.T
    [nrow, ncol] = decval_output.shape
    coverage = 0.0
    for i in range(nrow):
        max_rank_i = 0
        correct = np.where(groudtruth[i] == 1)[0]
        inv_ranks = np.argsort(decval_output[i])
        for j in correct:
            rank_j = len(inv_ranks) - np.where(inv_ranks == j)[0][0]
            max_rank_i = max(max_rank_i, rank_j)
        coverage += max_rank_i - 1
    coverage /= nrow
    return coverage

def Ranking_loss(decval_output, groudtruth, example_pivot=True):
    """
      Evaluate ranking loss in example pivot or label pivot mode.

      Arguments:
        decval_output:  decsion value matrix (example x label matrix).

        groudtruth: binary groudtruth matrix (example x label matrix).

        example_pivot:True if average precision is calculated in example-pivot, False
        if we calculate label-pivot average precision.

      Return:
        ranking loss.
    """
    # rank_loss_i = 1/|correct||incorrect| |rank_i(\l_a))>rank_i(\l_b)| where l_a in correct and l_b in incorrect.
    if (not example_pivot):
        decval_output = decval_output.T  # output is label x example
        groudtruth = groudtruth.T

    rloss = 0.0

    nrow = decval_output.shape[0]
    for i in range(nrow):
        correct = np.where(groudtruth[i] == 1)[0]
        incorrect = np.where(groudtruth[i] == 0)[0]
        inv_ranks = np.argsort(decval_output[i])
        if (len(correct) == 0 or len(incorrect) == 0):
            continue  # rank loss = 0

        nincorrect = 0.0
        for l_a in correct:
            for l_b in incorrect:
                rank_l_a = len(inv_ranks) - np.where(inv_ranks == l_a)[0][0]
                rank_l_b = len(inv_ranks) - np.where(inv_ranks == l_b)[0][0]
                if (rank_l_a > rank_l_b): nincorrect += 1
        rloss_i = nincorrect / (len(correct) * len(incorrect))
        rloss += rloss_i
    rloss /= nrow
    return rloss


def Average_precision(decval_output, groudtruth, example_pivot=True):
    """
      Evaluate average precision in example pivot or label pivot mode. In example-pivot:
      evaluate the average fraction of proper labels at positions where recall increases
      (one proper label is met.). In label-pivot: evaluate the average fraction of
      proper examples at positions where recall increases (one proper example is met.).

      Arguments:
        decval_output:  decsion value matrix (example x label matrix).

        groudtruth: binary groudtruth matrix (example x label matrix).

        example_pivot:True if average precision is calculated in example-pivot, False
        if we calculate label-pivot average precision.

      Return:
        average precision.
    """
    if (not example_pivot):
        decval_output = decval_output.T  # output is label x example
        groudtruth = groudtruth.T

    nrow = decval_output.shape[0]
    ap = 0.0
    nvalid = 0
    aps = []
    for i in range(nrow):
        gt_i = np.where(groudtruth[i] == 1)[0]
        if (len(gt_i) == 0): continue
        inv_ranks = np.argsort(decval_output[i])

        ap_i = 0.0
        ncorrect = 0.0
        for j in range(len(inv_ranks) - 1, -1, -1):
            l_j = inv_ranks[j]
            if (l_j in gt_i):
                ncorrect += 1
                ap_i += ncorrect / (len(inv_ranks) - j)
        ap_i /= len(gt_i)
        ap += ap_i
        aps.append(ap_i)
        nvalid += 1
    ap /= nrow
    # print 'nvalid', nvalid
    return ap


def get_metrics(y_true, y_pred):
    loss_name = ['hamming_loss','one_error','coverage','ranking_loss','ave_precision',
                 'microf1', 'macrof1','sub_acc']
    # example based
    hamming_loss = metrics.hamming_loss(y_true, y_pred)
    # example based
    one_error = metrics.zero_one_loss(y_true, y_pred)
    coverage = Coverage(y_pred, y_true, example_pivot=True)
    # coverage = metrics.coverage_error(y_true,y_pred)
    ranking_loss = Ranking_loss(y_pred, y_true)
    # ranking_loss = metrics.label_ranking_loss(y_true,y_true)
    # example based
    # ave_precision = metrics.average_precision_score(y_true, y_pred, average='macro')
    ave_precision = Average_precision(y_pred,y_true)
    micro_f1 = metrics.f1_score(y_true,y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true,y_pred, average='macro')
    subset_accuaracy = metrics.accuracy_score(y_true, y_pred)
    # jaccard_score = metrics.jaccard_similarity_score(y_true, y_pred)
    # label-based measures
    # macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
    # macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
    # micro_precision = metrics.precision_score(y_true, y_pred, average='micro')
    #  micro_recall = metrics.recall_score(y_true, y_pred, average='micro')
    loss_value = [hamming_loss,one_error,coverage,ranking_loss,ave_precision,micro_f1, macro_f1, subset_accuaracy]

    return dict(zip(loss_name,loss_value))

def print_result(result_list):
    import numpy as np
    print("{:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f}"
           .format(np.mean(result_list['hamming_loss']),np.std(result_list['hamming_loss']),
                   np.mean(result_list['one_error']),np.std(result_list['one_error']),
                   np.mean(result_list['coverage']),np.std(result_list['coverage']),
                   np.mean(result_list['ranking_loss']),np.std(result_list['ranking_loss']),
                   np.mean(result_list['ave_precision']),np.std(result_list['ave_precision']),
                   np.mean(result_list['microf1']),np.std(result_list['microf1']),
                   np.mean(result_list['macrof1']),np.std(result_list['macrof1']),
                   np.mean(result_list['sub_acc']),np.std(result_list['sub_acc']),
                   np.mean(result_list['time']),np.std(result_list['time'])))


