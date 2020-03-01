# Tutorial on Sample Generation

from src.datagenerator import XGenerator, Labels, Dataset
import numpy as np
import time
import unittest
import math


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, i)
    size = round(size_bytes / power, 2)
    return "{} {}".format(size, size_name[i])


len_mock = 100000
x_window = 1000
shape_sample = (100, 10)
mock_sequence = np.ones((len_mock,) + shape_sample)
size_memory = mock_sequence.__sizeof__()
print(f"Size of raw data is: {convert_size(size_memory)}")
size_for_rolling_window = size_memory * x_window
print(f"Size for rolling raw data would be: {convert_size(size_for_rolling_window)}")
start_index = np.datetime64(0, "ns")

mock_index = []
mock_index.append(start_index)
for i in range(1, len_mock):
    mock_index.append(mock_index[-1] + 1)
mock_index = np.asarray(mock_index)

x_exist = np.ones(mock_index.shape, dtype=np.bool)
x_non_exist = 30
x_exist[:x_non_exist] = False

multi_class_n = 2
mock_labels = np.repeat(np.arange(len_mock, dtype=np.int), multi_class_n).reshape((len_mock, multi_class_n))

x_generator = XGenerator(raw_sequence=mock_sequence,
                         raw_index=mock_index, x_window=x_window, x_exist=x_exist,
                         with_index=False)

if __name__ == "__main__":

    # performance tests
    samples = len(x_generator)
    start = time.clock()
    for i in range(samples):
        test_x = x_generator[i]
    duration = time.clock() - start
    print(f"Samples per second iteration:{samples / duration}")
    # ~ result should be ~100k-500k per second in run mode

    samples = len(x_generator)
    start = time.clock()
    for index in x_generator.index:
        test_x = x_generator[index]
    duration = time.clock() - start
    print(f"Samples per second with index:{samples / duration}")
    print(test_x.shape)
    # ~ result should be ~3.5k-4k per second in run mode

    stride = 5
    samples = int(len(x_generator) / stride)
    start = time.clock()
    # stride objct
    test_x = x_generator[::stride]
    duration = time.clock() - start
    print(f"Samples per second striding:{len(test_x) / duration}")
    print(test_x.shape)
    # ~ result should be ~3.5k-4k per second in run mode

    samples = int(len(x_generator) / 10)
    start = time.clock()
    # stride objct
    test_x = x_generator[range(samples)]
    duration = time.clock() - start
    print(f"Samples per second range:{samples / duration}")
    print(test_x.shape)
    # ~ result should be ~3.5k-4k per second in run mode

    # memory efficiency
    class TestMemory(unittest.TestCase):

        def test_memory(self):
            self.assertRaises(MemoryError, x_generator.__getitem__, slice(0, len(x_generator)))


    # get all x samples should raise a memory error
    TestMemory().test_memory()
