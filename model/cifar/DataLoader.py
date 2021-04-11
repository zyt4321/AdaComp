from __future__ import print_function
import os
import platform
import yaml
import pickle
import numpy
import random

class Dataloader:

    def __init__(self, data_path, sub_dataset):
        # 读取配置
        self.load_cifar10(data_path)
        self._split_train_valid(sub_dataset)
        self.n_train = self.train_images.shape[0]
        # self.n_valid = self.valid_images.shape[0]
        self.n_test = self.test_images.shape[0]
        print('\n' + '=' * 20 + ' load data ' + '=' * 20)
        print('# train data: %d' % (self.n_train))
        # print('# valid data: %d' % (self.n_valid))
        print('# test data: %d' % (self.n_test))
        print('=' * 20 + ' load data ' + '=' * 20 + '\n')

    # def _split_train_valid(self, valid_rate=0.9):
    def _split_train_valid(self, sub_dataset=1):
        images, labels = self.train_images, self.train_labels
        sub_size = images.shape[0] // sub_dataset
        sub_index = numpy.random.choice(images.shape[0], sub_size, replace=False)

        # thresh = int(images.shape[0] * valid_rate)
        self.train_images, self.train_labels = images[sub_index, :, :, :], labels[sub_index]
        # self.valid_images, self.valid_labels = images[thresh:, :, :, :], labels[thresh:]

    def load_cifar10(self, directory):
        # 读取训练集
        images, labels = [], []
        for filename in ['%s/data_batch_%d' % (directory, j) for j in range(1, 6)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
            for i in range(len(cifar10[b"labels"])):
                image = numpy.reshape(cifar10[b"data"][i], (3, 32, 32))
                image = numpy.transpose(image, (1, 2, 0))
                image = image.astype(float)
                images.append(image)
            labels += cifar10[b"labels"]
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        self.train_images, self.train_labels = images, labels

        # 读取测试集
        images, labels = [], []
        for filename in ['%s/test_batch' % (directory)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
            for i in range(len(cifar10[b"labels"])):
                image = numpy.reshape(cifar10[b"data"][i], (3, 32, 32))
                image = numpy.transpose(image, (1, 2, 0))
                image = image.astype(float)
                images.append(image)
            labels += cifar10[b"labels"]
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        self.test_images, self.test_labels = images, labels
