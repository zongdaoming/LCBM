#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# Author: zongdaoming
# Email: m15901768536@163.com
# Github: zongdaoming.github.io
# Datetime: 2018/11/11 
"""
# -*- coding: utf-8 -*-
import arff
import numpy as np
import os

from io import BytesIO
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer
from os.path import join
from collections import namedtuple
from skmultilearn.model_selection import IterativeStratification

# dir_path = os.path.dirname(os.path.realpath(__file__))
# dir_path = os.path.dirname(__file__)
dir_path = "/home/zongdaoming/LCBMr/LCBMr/jy_LCBM-master/"

# 分割数据集的参数，固定值，不需要修改
# n_splits = 20
# test_size = 0.30

n_splits = 10
test_size = 0.20
random_state = 2019

np.random.seed(2019)

MultiViewDataSet = namedtuple('MultiViewDataSet', ['X_view1', 'X_view2', 'y'])


def split_dataset2(X_view1, X_view2, y):
    kf = IterativeStratification(n_splits=5, order=1)
    for train_index, test_index in kf.split(X_view1, y):
        train = MultiViewDataSet(
            X_view1[train_index],
            X_view2[train_index],
            y[train_index]
        )
        test = MultiViewDataSet(
            X_view1[test_index],
            X_view2[test_index],
            y[test_index]
        )
        yield train, test


def split_dataset(X_view1, X_view2, y):
    ss = ShuffleSplit(
        n_splits=n_splits, test_size=test_size,
        random_state=random_state
    )
    for index in ss.split(X_view1, y):
        train_index, test_index = index

        train = MultiViewDataSet(
            X_view1[train_index],
            X_view2[train_index],
            y[train_index]
        )
        test = MultiViewDataSet(
            X_view1[test_index],
            X_view2[test_index],
            y[test_index]
        )
        yield train, test


def save_dataset_by_format(fpath, fname_format, dataset):
    dataset_dict = dataset._asdict()
    for key in dataset_dict:
        fname = fname_format.format(key)
        np.savetxt(join(fpath, fname), dataset_dict[key])


def load_from_arff(filename, labelcount, input_feature_type='float', encode_nominal=True):
    arff_frame = arff.load(
        open(filename, 'r'),
        encode_nominal=encode_nominal
    )
    matrix = np.array(
        arff_frame['data'],
        dtype=input_feature_type
    )
    X, y = matrix[:, :-labelcount], matrix[:, -labelcount:].astype(int)
    return X, y


def load_from_svmlight_file(filename, offset, n_features, classes):
    lines = [line[offset:] for line in open(filename, 'r')]
    bio = BytesIO(os.linesep.join(lines).encode('utf-8'))

    X, y = load_svmlight_file(bio, multilabel=True, n_features=n_features)
    X = X.toarray()
    y = MultiLabelBinarizer(classes=classes).fit_transform(y)
    return X, y


def load_scene():
    filename = join(dir_path, 'datasets', 'scene', 'scene.arff')
    X, y = load_from_arff(filename=filename, labelcount=6)
    view1_index = np.arange(0, 98)
    view2_index = np.arange(98, 294)
    X_view1 = X[:, view1_index]
    X_view2 = X[:, view2_index]
    return (X_view1, X_view2), y


def load_emotions():
    filename = join(dir_path, 'datasets', 'emotions', 'emotions.arff')
    X, y = load_from_arff(filename=filename, labelcount=6)
    # View 1:
    # 8 rhythmic attributes obtained by extracting
    # periodic changes from a beat histogram
    view1_index = np.arange(64, 72)
    X_view1 = X[:, view1_index]

    # View 2:
    # 64 timbre attributes obtained from the mel frequency cepstral coefficients (MFCCs)
    # and from the spectral centroid, spectral rolloff, and spectral flux-extracted
    # short-term Fourier transform (FFT)
    view2_index = np.arange(0, 64)
    X_view2 = X[:, view2_index]

    return (X_view1, X_view2), y


def load_human():
    train_filename = join(dir_path, 'datasets', 'HumanPseAAC', 'HumanPseAAC3106-train.txt')
    test_filename = join(dir_path, 'datasets', 'HumanPseAAC', 'HumanPseAAC3106-test.txt')

    X_train, y_train = load_from_svmlight_file(
        train_filename, offset=7,
        n_features=440, classes=range(1, 15)
    )
    X_test, y_test = load_from_svmlight_file(
        test_filename, offset=7,
        n_features=440, classes=range(1, 15)
    )

    # Concatenate training and testing dataset
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # View 1:
    # 20 amino acid and 20 pseudo-amino acid
    view1_index = np.arange(0, 40)
    X_view1 = X[:, view1_index]

    # View 2:
    # 400 diptide components
    view2_index = np.arange(40, 440)
    X_view2 = X[:, view2_index]
    return (X_view1, X_view2), y


def load_plant():
    train_filename = join(dir_path, 'datasets', 'PlantPseAAC', 'PlantPseAAC978-train.txt')
    test_filename = join(dir_path, 'datasets', 'PlantPseAAC', 'PlantPseAAC978-test.txt')

    X_train, y_train = load_from_svmlight_file(
        train_filename, offset=7,
        n_features=440, classes=range(1, 13)
    )
    X_test, y_test = load_from_svmlight_file(
        test_filename, offset=7,
        n_features=440, classes=range(1, 13)
    )

    # Concatenate training and testing dataset
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # View 1:
    # 20 amino acid and 20 pseudo-amino acid
    view1_index = np.arange(0, 40)
    X_view1 = X[:, view1_index]

    # View 2:
    # 400 diptide components
    view2_index = np.arange(40, 440)
    X_view2 = X[:, view2_index]
    return (X_view1, X_view2), y
