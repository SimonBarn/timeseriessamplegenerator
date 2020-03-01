import numpy as np
import time
from src.datagenerator import Labels, XGenerator, Dataset, BatchGenerator
from keras.models import Model as KerasModel
from keras.layers import Input, Dense

len_mock = 100000
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

    batch_size = 32
    batch_generator = BatchGenerator(dataset=dataset, batch_size=32, shuffle=True)

    samples = len(batch_generator)
    start = time.clock()
    for batch in batch_generator:
        test_batch = batch
    duration = time.clock() - start
    print(f"Samples per second:{samples * batch_size / duration}")
    print(test_batch[0].shape, test_batch[1].shape)
    print(f"Batches per second:{samples / duration}")
    # samples ~ 7-10k / s
    # batches ~ 250-350 /s

    batch_size = 32
    batch_generator = BatchGenerator(dataset=dataset, batch_size=32, shuffle=False)

    samples = len(batch_generator)
    start = time.clock()
    for batch in batch_generator:
        test_batch = batch
    duration = time.clock() - start
    print(f"Samples per second:{samples * batch_size / duration}")
    print(test_batch[0].shape, test_batch[1].shape)
    print(f"Batches per second:{samples / duration}")
    # samples ~ 13k / s
    # batches ~ 400 /s

    batch_size = 32
    batch_generator = BatchGenerator(dataset=dataset, batch_size=32, shuffle=True, oversampling=True)

    samples = len(batch_generator)
    start = time.clock()
    for batch in batch_generator:
        test_batch = batch
    duration = time.clock() - start
    print(f"Samples per second:{samples * batch_size / duration}")
    print(test_batch[0].shape, test_batch[1].shape)
    print(f"Batches per second:{samples / duration}")
    # samples ~ 10k / s
    # batches ~ 350 /s

    batch_size = 32
    batch_generator = BatchGenerator(dataset=dataset, batch_size=32, stride=2)

    samples = len(batch_generator)
    start = time.clock()
    for batch in batch_generator:
        test_batch = batch
    duration = time.clock() - start
    print(f"Samples per second:{samples * batch_size / duration}")
    print(test_batch[0].shape, test_batch[1].shape)
    print(f"Batches per second:{samples / duration}")
    # samples ~ 13k / s
    # batches ~ 400 /s

    batch_size = 32
    batch_generator = BatchGenerator(dataset=dataset, batch_size=32, stride=2)

    samples = len(batch_generator)
    start = time.clock()
    for _ in range(len(batch_generator)):
        batch = next(batch_generator)
    duration = time.clock() - start
    print(f"Samples per second:{samples * batch_size / duration}")
    print(test_batch[0].shape, test_batch[1].shape)
    print(f"Batches per second:{samples / duration}")
    # samples ~ 12-13k / s
    # batches ~ 350-400 /s
