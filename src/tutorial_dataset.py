import numpy as np
import time
from src.datagenerator import Labels, XGenerator, Dataset

len_mock = 1000
x_window = 10

shape_sample = (100, 10)
mock_sequence = np.ones((len_mock,) + shape_sample)
start_index = np.datetime64(0, "ns")
mock_index = []
mock_index.append(start_index)
for i in range(1, len_mock):
    mock_index.append(mock_index[-1] + 1)
mock_index = np.asarray(mock_index)

mock_labels = np.asarray([[np.random.randint(2), np.random.randint(2)] for _ in range(len_mock)])
x_exist = np.ones(mock_index.shape, dtype=np.bool)
x_non_exist = 30
x_exist[:x_non_exist] = False

if __name__ == "__main__":
    labels = Labels(labels=mock_labels, index=mock_index, with_index=False)
    x_generator = XGenerator(raw_sequence=mock_sequence,
                             raw_index=mock_index, x_window=x_window, x_exist=x_exist,
                             with_index=False)
    dataset = Dataset(x=x_generator, y=labels, name="tutorial")
    samples = len(dataset)
    start = time.clock()
    for i in range(samples):
        test = dataset[i]
    duration = time.clock() - start
    print(f"Samples per second:{samples / duration}")
    # in run mode ~ 50k - 180k /s

    start = time.clock()
    for i in range(samples):
        test = dataset.get_shuffled_sample()
    duration = time.clock() - start
    print(f"Samples per second:{samples / duration}")
    # in run mode ~ 30k - 50k /s

    start = time.clock()
    for i in range(samples):
        test = dataset.get_oversampled_sample()
    duration = time.clock() - start
    print(f"Samples per second:{samples / duration}")
    # in run mode ~ 4k - 7k /s

    class_choice = [1, 0]
    start = time.clock()
    for i in range(samples):
        test = dataset.get_shuffled_sample_from_specified_class(class_choice)
    duration = time.clock() - start
    print(f"Samples per second:{samples / duration}")
    # in run mode ~ 14k - 19k /s

    unique_classes_list, unique_classes_n_samples = dataset.get_count_classes()
    unique_classes_list = dataset.get_unique_classes()

    print(dataset.class_count)
    print(dataset.y_shape)
    print(dataset.name)
    print(dataset.index[:5])
    print(dataset.x_shape)
