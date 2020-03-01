from unittest import TestCase
from src.datagenerator import XGenerator, Labels, Dataset, BatchGenerator
import numpy as np

len_mock = 1000
x_window = 10

mock_prices = np.arange(len_mock, dtype=np.int)

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


class TestXGenerator(TestCase):
    def setUp(self) -> None:
        self.x_generator = XGenerator(raw_index=mock_index, raw_sequence=mock_prices, x_window=x_window,
                                      x_exist=x_exist)

    def test_init_errors(self):
        self.assertRaises(AssertionError, XGenerator, raw_index=mock_index, raw_sequence=mock_prices, x_window=x_window,
                          x_exist=x_exist[10:])
        self.assertRaises(AssertionError, XGenerator, raw_index=mock_index, raw_sequence=mock_prices[10:],
                          x_window=x_window, x_exist=x_exist)
        self.assertRaises(AssertionError, XGenerator, raw_index=mock_index[10:], raw_sequence=mock_prices,
                          x_window=x_window, x_exist=x_exist)

    def test_returns(self):
        for index in range(0, len_mock - x_non_exist):
            assert np.array_equal(self.x_generator[index],
                                  np.arange(x_non_exist - x_window + 1 + index, x_non_exist + 1 + index))
        index = len_mock - x_non_exist
        assert not np.array_equal(self.x_generator[1],
                                  np.arange(x_non_exist - x_window + 1 + index, x_non_exist + 1 + index))
        assert len(self.x_generator) == len_mock - x_non_exist
        for i in range(len(self.x_generator)):
            assert self.x_generator[i][0] + 1 == self.x_generator[i][1]
        assert np.array_equal(self.x_generator[mock_index[-1]], mock_prices[list(range(-10, 0))])
        assert np.array_equal(self.x_generator[-1], mock_prices[list(range(-10, 0))])

    def test_index(self):
        assert self.x_generator.index[0] == mock_index[x_non_exist]
        assert self.x_generator.index[-1] == mock_index[-1]
        assert np.array_equal(self.x_generator.index, mock_index[x_non_exist:])

    def test_returns_with_index(self):
        self.x_generator.with_index = True
        assert np.array_equal(self.x_generator[mock_index[-1]][1],
                              mock_prices[[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]])
        assert np.array_equal(self.x_generator[mock_index[-1]][0], mock_index[-1])
        assert isinstance(self.x_generator[0], tuple)
        assert isinstance(self.x_generator[0][0], np.datetime64)
        assert self.x_generator[0][0] == mock_index[x_non_exist]
        assert self.x_generator[-1][0] == mock_index[-1]
        assert self.x_generator.shape == (
            mock_index[x_non_exist:].shape, (mock_prices[x_non_exist:].shape[0], x_window))
        for index in range(0, len_mock - x_non_exist):
            assert np.array_equal(self.x_generator[index][0], mock_index[x_non_exist:][index])
            assert np.array_equal(self.x_generator[index][1],
                                  np.arange(x_non_exist - x_window + 1 + index, x_non_exist + 1 + index))
        self.x_generator.with_index = False
        assert isinstance(self.x_generator[0], np.ndarray)

    def test_get_item(self):
        self.x_generator.with_index = False
        assert len(self.x_generator[:20]) == 20
        assert len(self.x_generator.index[::3]) == len(self.x_generator) // 3 + 1
        assert len(self.x_generator.index[range(10)]) == 10
        assert len(self.x_generator.index[range(0, 10, 2)]) == 5
        assert len(self.x_generator.index[range(0, 10, 3)]) == 4

    def test_data_efficiency(self):
        assert self.x_generator._x_windowed.__sizeof__() < mock_prices.__sizeof__()

    def test_append(self):
        gen_1 = XGenerator(raw_index=mock_index[:50], raw_sequence=mock_prices[:50], x_exist=x_exist[:50],
                           x_window=x_window)
        gen_2 = XGenerator(raw_index=mock_index[50:], raw_sequence=mock_prices[50:], x_exist=x_exist[50:],
                           x_window=x_window)
        gen_1.append(gen_2)
        assert len(gen_1) == len(self.x_generator)
        for i in range(len(self.x_generator)):
            assert np.array_equal(gen_1[i], self.x_generator[i])
        assert np.array_equal(gen_1.index, self.x_generator.index)


class TestLabels(TestCase):
    def setUp(self) -> None:
        self.labels = Labels(labels=mock_labels, index=mock_index, with_index=False)

    def test_init_error(self):
        self.assertRaises(AssertionError, Labels, mock_labels[:10], mock_index)

    def test_shape(self):
        assert self.labels.shape == (len(mock_index), mock_labels.shape[-1])

    def test_len(self):
        assert len(self.labels) == len(mock_index) == len(mock_labels)

    def test_returns(self):
        for i, item in enumerate(mock_labels):
            assert np.array_equal(self.labels[i], mock_labels[i])
        self.labels.with_index = True
        for i, (index, item) in enumerate(zip(mock_index, mock_labels)):
            assert np.array_equal(self.labels[i][0], index)
            assert np.array_equal(self.labels[i][1], item)
        for i, index in enumerate(mock_index):
            assert np.array_equal(self.labels[i][1][0], index.astype(int))
            assert np.array_equal(self.labels[i][1][1], index.astype(int))
        for i, index in enumerate(mock_index):
            assert np.array_equal(self.labels[index][1][0], index.astype(int))
            assert np.array_equal(self.labels[index][1][1], index.astype(int))
        self.labels.with_index = False

    def test_append(self):
        y_1 = Labels(labels=mock_labels[:50], index=mock_index[:50])
        y_2 = Labels(labels=mock_labels[50:], index=mock_index[50:])
        y_1.append(y_2)
        for i in range(len(self.labels)):
            assert np.array_equal(y_1[i], self.labels[i])
        assert len(y_1) == len(self.labels)
        assert y_1.shape == self.labels.shape
        assert np.array_equal(y_1.index, self.labels.index)


class TestDatasetEmpty(TestCase):
    def setUp(self) -> None:
        self.x_generator = XGenerator(raw_index=mock_index + len_mock, raw_sequence=mock_prices, x_window=x_window,
                                      x_exist=x_exist)
        self.labels = Labels(labels=mock_labels, index=mock_index, with_index=False)
        self.dataset = Dataset(x=self.x_generator, y=self.labels, name="test")

    def test_len(self):
        assert len(self.dataset) == 0
        self.assertRaises(IndexError, self.dataset.__getitem__, 0)
        self.assertRaises(ValueError, self.dataset.get_shuffled_sample)
        assert np.sum(self.dataset.class_count) == 0
        assert len(self.dataset.unique_classes) == len_mock


class TestDatasetPlus1(TestCase):
    def setUp(self) -> None:
        self.plus_index = 1
        self.x_generator = XGenerator(raw_index=mock_index + self.plus_index, raw_sequence=mock_prices,
                                      x_window=x_window, x_exist=x_exist)
        self.labels = Labels(labels=mock_labels, index=mock_index, with_index=False)
        self.dataset = Dataset(x=self.x_generator, y=self.labels, name="test")

    def test_len(self):
        assert len(self.dataset) == len_mock - self.plus_index - np.sum(~x_exist)

    def test_returns(self):
        assert np.array_equal(self.dataset[0][0], self.x_generator[0])
        assert np.array_equal(self.dataset[0][1], self.labels[self.plus_index + np.sum(~x_exist)])
        assert np.array_equal(self.dataset[-1][0], self.x_generator[-self.plus_index - 1])
        assert np.array_equal(self.dataset[-1][1], self.labels[-1])

    def test_n_samples(self):
        unique_classes, class_clount = self.dataset.get_count_classes()
        assert np.sum(class_clount) == len(self.dataset)

    def test_get(self):
        shuffled_sample = self.dataset.get_shuffled_sample()
        assert np.array_equal(shuffled_sample[0][-1] + 1, shuffled_sample[1][0])
        unique_classes_2, class_clount = self.dataset.get_count_classes()
        class_choice = unique_classes_2[-1]
        sample = self.dataset.get_shuffled_sample_from_specified_class(class_choice)
        assert np.array_equal(sample[1], class_choice)
        unique_classes_2, class_clount = self.dataset.get_count_classes()
        class_choice = unique_classes_2[0]
        self.assertRaises(ValueError, self.dataset.get_shuffled_sample_from_specified_class, class_choice)


class TestDatasetSame(TestCase):
    def setUp(self) -> None:
        self.plus_index = 0
        x_exist_new = np.ones(len_mock, np.bool)
        x_window = 1
        self.x_generator = XGenerator(raw_index=mock_index + self.plus_index, raw_sequence=mock_prices,
                                      x_window=x_window, x_exist=x_exist_new)
        self.labels = Labels(labels=mock_labels, index=mock_index, with_index=False)
        self.dataset = Dataset(x=self.x_generator, y=self.labels, name="test")

    def test_returns(self):
        for i in range(100):
            sample = self.dataset.get_shuffled_sample()
            x_sample = sample[0]
            y_sample = sample[1]
            assert x_sample[-1] == y_sample[0] == y_sample[1]
        for i in range(100):
            sample = self.dataset.get_oversampled_sample()
            x_sample = sample[0]
            y_sample = sample[1]
            assert x_sample[-1] == y_sample[0] == y_sample[1]


class TestDatasetPlusAndPrice(TestCase):
    def setUp(self) -> None:
        self.plus_index = 1
        self.x_generator = XGenerator(raw_index=mock_index + self.plus_index,
                                      raw_sequence=mock_prices + self.plus_index, x_window=x_window + 10,
                                      x_exist=x_exist)
        self.labels = Labels(labels=mock_labels, index=mock_index, with_index=False)
        self.dataset = Dataset(x=self.x_generator, y=self.labels, name="test")

    def test_returns(self):
        for i in range(100):
            sample = self.dataset.get_shuffled_sample()
            x_sample = sample[0]
            y_sample = sample[1]
            assert x_sample[-1] == y_sample[0] == y_sample[1]


class TestDatasetoversampling(TestCase):
    def setUp(self) -> None:
        self.class_high = [0, 0]
        self.class_low = [1, 1]
        mock_labels_2 = np.asarray([self.class_high for _ in range(len_mock)])
        mock_labels_2[50] = self.class_low
        self.x_generator = XGenerator(raw_index=mock_index,
                                      raw_sequence=mock_prices, x_window=x_window,
                                      x_exist=x_exist)
        self.labels = Labels(labels=mock_labels_2, index=mock_index, with_index=False)
        self.dataset = Dataset(x=self.x_generator, y=self.labels, name="test")

    def test_oversampling(self):
        class_high_count = 0
        class_low_count = 0
        for i in range(2000):
            x, y = self.dataset.get_oversampled_sample()
            if np.array_equal(y, self.class_high):
                class_high_count += 1
            if np.array_equal(y, self.class_low):
                class_low_count += 1
        assert 0.8 < class_high_count / class_low_count < 1.2


class TestBatchGenerator(TestCase):

    def setUp(self) -> None:
        len_mock = 1000
        x_window = 10

        self.mock_prices = np.arange(len_mock, dtype=np.int)

        start_index = np.datetime64(0, "ns")
        mock_index = []
        mock_index.append(start_index)
        for i in range(1, len_mock):
            mock_index.append(mock_index[-1] + 1)
        self.mock_index = np.asarray(mock_index)

        x_exist = np.ones(self.mock_index.shape, dtype=np.bool)
        self.x_non_exist = 30
        x_exist[:x_non_exist] = False
        self.batch_size = 2
        multi_class_n = 2
        self.mock_labels = np.repeat(np.arange(len_mock, dtype=np.int), multi_class_n).reshape(
            (len_mock, multi_class_n))

        self.x_generator = XGenerator(raw_index=self.mock_index,
                                      raw_sequence=self.mock_prices, x_window=x_window,
                                      x_exist=x_exist)
        self.labels = Labels(labels=self.mock_labels, index=self.mock_index, with_index=False)
        self.dataset = Dataset(x=self.x_generator, y=self.labels, name="test")
        self.batch_generator = BatchGenerator(dataset=self.dataset, batch_size=self.batch_size, shuffle=False,
                                              oversampling=False,
                                              iterations=None, stride=None, x_only=False)

    def test_get_batch_next(self):
        batch = next(self.batch_generator)
        x_batch = batch[0]
        y_batch = batch[1]
        assert np.array_equal(x_batch[0][-1], self.mock_prices[self.x_non_exist])
        assert np.array_equal(x_batch[0][-2], self.mock_prices[self.x_non_exist - 1])
        assert np.array_equal(y_batch[0][0], self.mock_labels[self.x_non_exist][0])
        assert np.array_equal(y_batch[0][1], self.mock_labels[self.x_non_exist][1])

    def test_get_batch_iter(self):

        for i, batch in enumerate(self.batch_generator):
            x_batch = batch[0]
            y_batch = batch[1]
            assert np.array_equal(x_batch[0][-1], self.mock_prices[self.x_non_exist + i * self.batch_size])
            assert np.array_equal(x_batch[0][-2], self.mock_prices[self.x_non_exist - 1 + i * self.batch_size])
            assert np.array_equal(y_batch[0][0], self.mock_labels[self.x_non_exist + i * self.batch_size][0])
            assert np.array_equal(y_batch[0][1], self.mock_labels[self.x_non_exist + i * self.batch_size][1])

    def test_stop_iteration(self):
        batch_generator = BatchGenerator(dataset=self.dataset, batch_size=1, shuffle=False, oversampling=False,
                                         iterations=None, stride=None, x_only=False)

        stopped = False
        for n in range(len(self.dataset)):
            try:
                batch = next(batch_generator)
            except StopIteration:
                stopped = True
        assert not stopped

        batch_generator = BatchGenerator(dataset=self.dataset, batch_size=1, shuffle=False, oversampling=False,
                                         iterations=None, stride=None, x_only=False)

        stopped = False
        for n in range(len(self.dataset) + 1):
            try:
                batch = next(batch_generator)
            except StopIteration:
                stopped = True
        assert stopped
