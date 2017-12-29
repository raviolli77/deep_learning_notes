
# coding: utf-8

# # Data Extraction
# 
# Heavily inspired by [Package Docs](https://github.com/sorki/python-mnist/blob/master/mnist/loader.py) along with answers from stack overflow. 

import struct
import numpy as np
from array import array


def load_labels(file_name):
    with open(file_name, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        set_array = array('B', f.read())
    return set_array


def load_images(file_name):
    with open(file_name, 'rb') as f:
        magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
        set_array = array('B', f.read())
    return size, rows, cols, set_array

class_set_array = load_labels('train-labels-idx1-ubyte')


class_set = np.array(class_set_array)

size, rows, cols, training_set_array = load_images('train-images-idx3-ubyte')


training_set = np.array(training_set_array).reshape(size, rows, cols)


# Test Set Extraction
size_test, rows_test, cols_test, test_set_array = load_images('t10k-images-idx3-ubyte')

test_set = np.array(test_set_array).reshape(size_test, rows_test, cols_test)


class_test_set = load_labels('t10k-labels-idx1-ubyte')

class_test_set = np.array(class_test_set)

