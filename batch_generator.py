#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np


class BatchGenerator(object):
    def __init__(self, views_1, views_2, labels):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._views_1 = views_1
        self._views_2 = views_2
        self._labels = labels
        self._num_examples = views_1.shape[0]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:  # start a new round of traversal
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)  # arange函数用于创建等差数组
            np.random.shuffle(perm)  # shuffle the data
            self._views_1 = self._views_1[perm]
            self._views_2 = self._views_2[perm]
            self._labels = self._labels[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._views_1[start:end], self._views_2[start:end], self._labels[start:end]


class DataGenerator(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(x)

    def next_batch(self, batch_size, faka_data=False, shuffle=True):
        """Return the next 'batch_size' examples from this data set"""
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._x = self._x[perm]
            self._y = self._y[perm]
        # Go to next batch
        if start + batch_size > self._num_examples:
            """End epoch and restart a new epoch"""
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start  # 最后不够一个batch还剩下几个
            x_rest_part = self._x[start:self._num_examples]
            y_rest_part = self._y[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._x = self._x[perm]
                self._y = self._y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            x_new_part = self._x[start:end]
            y_new_part = self._y[start:end]
            return np.concatenate((x_rest_part, x_new_part), axis=0), \
                   np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size  # start = index_in_epoch
            end = self._index_in_epoch  # end is easy,index_in_epoch + batch_size
            return self._x[start:end], self._y[start:end]  # return x, y


class Dataset(object):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(images)

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch  # self._index_in_epoch  所有的调用，总共用了多少个样本，相当于一个全局变量
        #  #start第一个batch为0，剩下的就和self._index_in_epoch一样，如果超过了一个epoch，在下面还会重新赋值
        # Shuffle for the first epoch 第一个epoch需要shuffle
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)  # 生成一个样本长度的np.array
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
        # Go to next batch
        if start + batch_size > self._num_examples:  # epoch的结尾和下一个epoch的开头
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start  # 最后不够一个batch还剩下几个
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:  # 除了第一个epoch，以及每个epoch的开头，剩下中间batch的处理方式
            self._index_in_epoch += batch_size  # start = index_in_epoch
            end = self._index_in_epoch  # end很简单，就是 index_in_epoch加上batch_size
            return self._images[start:end], self._labels[start:end]  # 在数据x,y
