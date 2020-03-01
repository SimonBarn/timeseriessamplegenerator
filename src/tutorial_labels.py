import numpy as np
import time
from src.datagenerator import Labels

len_mock = 1000
x_window = 10

start_index = np.datetime64(0, "ns")
mock_index = []
mock_index.append(start_index)
for i in range(1, len_mock):
    mock_index.append(mock_index[-1] + 1)
mock_index = np.asarray(mock_index)

multi_class_n = 2
mock_labels = np.repeat(np.arange(len_mock, dtype=np.int), multi_class_n).reshape((len_mock, multi_class_n))

if __name__ == "__main__":

    labels = Labels(labels=mock_labels, index=mock_index, with_index=False)

    samples = len(labels)
    start = time.clock()
    for i in range(samples):
        test_sample = labels[i]
    duration = time.clock() - start
    print(f"Samples per second:{samples / duration}")
    # in run mode ~ 500k - 1000k /s

    labels = Labels(labels=mock_labels, index=mock_index, with_index=True)

    samples = len(labels)
    start = time.clock()
    for i in range(samples):
        test_sample = labels[i]
    duration = time.clock() - start
    print(f"Samples per second:{samples / duration}")
    # in run mode ~ 200k - 600k /s

    labels = Labels(labels=mock_labels, index=mock_index, with_index=True)

    samples = len(labels)
    start = time.clock()
    for index in labels.index:
        test_sample = labels[index]
    duration = time.clock() - start
    print(f"Samples per second:{samples / duration}")
    # in run mode ~ 30k - 70k /s

    labels_2 = Labels(labels=mock_labels, index=mock_index + len_mock, with_index=True)

    labels.append(labels_2)
    samples = len(labels)
    start = time.clock()
    for index in labels.index:
        test_sample = labels[index]
    duration = time.clock() - start
    print(f"Samples per second:{samples / duration}")
    # in run mode ~ 30k - 70k /s
