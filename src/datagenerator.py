import datetime
import numpy as np
import logging
import keras
from numbers import Number
import copy

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s:%(name)s:%(threadName)s:%(thread)s:%(levelname)s: %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class XGenerator:
    """
    This class behaves like a numpy array, which stores the x samples for the user. In the inside it creates rolling
    windows from a raw data sequence on the fly to lower the memory usage for samples, which are rolling samples.

    """

    def __init__(self, raw_index: np.ndarray, raw_sequence: np.ndarray, x_window: int, x_exist: np.ndarray,
                 with_index: bool = False):
        """

        :param raw_index: the index of the underlying raw sequence.
        :param raw_sequence: the raw data sequence.
        :param x_window: the rolling window
        :param x_exist: for each index we need a bool if the x exist here or not. This can be used to prohibt
                        the rolling window from create samples from data of different days.
                        Example:
                        raw_index = [datetime.datetime(year = 2020, month = 10, day = 1, hour = 22),
                                    datetime.datetime(year = 2020, month = 10, day = 1, hour = 23),
                                    datetime.datetime(year = 2020, month = 10, day = 2, hour = 8),
                                    datetime.datetime(year = 2020, month = 10, day = 2, hour = 9)]
                        raw_sequence = [10,11,20,21]
                        x_window = 2
                        x_exist = [False, True, False, True]
                        print(self[0]) -> [22,23]
                        print(self[1]) -> [20,21]
                        print(self[2]) -> error, out of index
        :param with_index: If true, the object returns a tuple with (index[i], x[i]), if false it return only x[i]
        """
        self._raw_index = raw_index
        self._raw_sequence = raw_sequence
        self.x_window = x_window
        self._x_exist_orig = x_exist
        # the first x_window -1 indices are needed to produce the first x window.
        self._x_exist = copy.deepcopy(x_exist)
        self._x_exist[:x_window - 1] = False
        self.with_index = with_index
        assert len(raw_index) == len(raw_sequence) == len(x_exist)
        shape = (self._raw_sequence.shape[0] - x_window + 1, x_window) + self._raw_sequence.shape[1:]
        strides = (self._raw_sequence.strides[0], self._raw_sequence.strides[0]) + self._raw_sequence.strides[1:]
        self._x_windowed = np.lib.stride_tricks.as_strided(self._raw_sequence, shape=shape, strides=strides)
        self._x_exist_windowed = self._x_exist[self.x_window - 1:]

        # translates self.index to index of x_windowed
        self._index_int_x_windowed_map = np.arange(len(self._x_exist[self.x_window - 1:]), dtype=np.int)[
            self._x_exist_windowed]
        self.index = self._raw_index[self._x_exist]

    def __getitem__(self, index) -> np.array:
        if isinstance(index, slice) or isinstance(index, range):
            int_windowed_x_indices = self._index_int_x_windowed_map[index]
            if self.with_index:
                return self.index[index], np.take(self._x_windowed, int_windowed_x_indices)
            else:
                return self._x_windowed[int_windowed_x_indices]
        elif isinstance(index, list):
            return self[np.asarray(index)]

        elif isinstance(index, Number):
            if self.with_index:
                return self.index[index], self._x_windowed[self._translate_index_to_windowed_index(index)]
            else:
                return self._x_windowed[self._translate_index_to_windowed_index(index)]

        elif isinstance(index, np.ndarray) and (index.dtype == int or index.dtype == bool):
            int_windowed_x_indices = self._index_int_x_windowed_map[index]
            if self.with_index:
                return self.index[index], np.take(self._x_windowed, int_windowed_x_indices)
            else:
                return self._x_windowed[int_windowed_x_indices]

        elif isinstance(index, datetime.datetime) or isinstance(index, np.datetime64):
            index_int = np.argwhere(self.index == index)
            if len(index_int) == 1:
                index_x_windowed = self._translate_index_to_windowed_index(index_int[0][0])
                if self.with_index:
                    return index, self._x_windowed[index_x_windowed]
                else:
                    return self._x_windowed[index_x_windowed]
            elif len(index_int) == 0:
                msg = f"No X value found for this index {index}"
                logger.error(msg)
                raise KeyError(msg)
            else:
                msg = "Duplicated index. The Index must be unique."
                logger.error(msg)
                raise KeyError(msg)

        else:
            msg = f"Index type {type(index)} not implemented"
            logger.error(msg)
            raise NotImplementedError(msg)

    @staticmethod
    def _translate_slice_to_range(slice_object, target_array) -> range:
        indices = slice_object.indices(len(target_array))
        indices_range = range(indices[0], indices[1], indices[2])
        return indices_range

    def _translate_index_to_windowed_index(self, index) -> int:
        index_x_windowed = self._index_int_x_windowed_map[index]
        return index_x_windowed

    def __len__(self):
        return np.sum(self._x_exist)

    @property
    def shape(self) -> tuple:
        if self.with_index:
            return (self.index.shape, (len(self),) + (self.x_window,) + (self._raw_sequence.shape[1:]))
        else:
            return (len(self),) + (self.x_window,) + (self._raw_sequence.shape[1:])

    def append(self, other) -> None:
        if self.x_window != other.x_window:
            msg = "Try to append X Generators with different x_windows. You can only append " \
                  "Generators with the same x_window"
            logger.error(msg)
            raise AssertionError(msg)
        if np.sum(np.isin(self._raw_index, other._raw_index)) != 0:
            msg = "The Index must be unique."
            logger.error(msg)
            raise AssertionError(msg)
        else:
            x_exist_new = np.concatenate([self._x_exist_orig, other._x_exist_orig])
            raw_index_new = np.concatenate([self._raw_index, other._raw_index])
            raw_sequence_new = np.concatenate([self._raw_sequence, other._raw_sequence])
            self = self.__init__(raw_index=raw_index_new, raw_sequence=raw_sequence_new, x_window=self.x_window,
                                 x_exist=x_exist_new, with_index=self.with_index)


def split_X_generator(x_generator: XGenerator, how: str, index: datetime.datetime) -> XGenerator:
    """

    :param x_generator: A XGenerator object
    :param how: Accepts: "<", "<=",  ">" and ">=" as strings
    :param index: the index, it should be compared against/splitted
    :return: Instance of XGenerator
    """
    if how == "<":
        mask = x_generator._raw_index < index
    elif how == "<=":
        mask = x_generator._raw_index <= index
    elif how == ">":
        mask = x_generator._raw_index > index
    elif how == ">=":
        mask = x_generator._raw_index >= index
    else:
        msg = f"The how '{how}' is not implemented."
        logger.error(msg)
        raise NotImplementedError(msg)

    index_raw_new = x_generator._raw_index[mask]
    raw_series_new = x_generator._raw_sequence[mask]
    x_exist_new = x_generator._x_exist[mask]

    return XGenerator(raw_index=index_raw_new, raw_sequence=raw_series_new, x_window=x_generator.x_window,
                      x_exist=x_exist_new, with_index=x_generator.with_index)


class Labels:
    """
    A Object, which structures the labels for an ML experiment.
    """

    def __init__(self, labels: np.ndarray, index: np.ndarray, with_index: bool = False):
        """

        :param labels: the labels/classes
        :param index: the index of the labels
        :param with_index: if it should be returned with the index or without
        """
        self.index = index
        self.labels = labels
        self.with_index = with_index
        assert len(self.index) == len(self.labels)

    def __getitem__(self, index) -> np.array:
        if isinstance(index, datetime.datetime) or isinstance(index, np.datetime64):
            index_int = np.argwhere(self.index == index)
            if len(index_int) == 1:
                if self.with_index:
                    return index, self.labels[index_int[0][0]]
                else:
                    return self.labels[index_int[0][0]]
            else:
                msg = f"Duplicated Index! {index}"
                logger.error(msg)
                raise ValueError(msg)
        else:
            if self.with_index:
                return self.index[index], self.labels[index]
            else:
                return self.labels[index]

    def append(self, other):
        new_index = np.concatenate([self.index, other.index])
        new_labels = np.concatenate([self.labels, other.labels])
        self = self.__init__(labels=new_labels, index=new_index, with_index=self.with_index)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def shape(self) -> tuple:
        if self.with_index:
            return (self.index.shape, self.labels.shape)
        else:
            return self.labels.shape


class Dataset:
    """
    A class, which represents a Dataset to train a ML Model. It contains a set of samples. Each Sample contains X, y and
    an index. This Dataset can efficiently random sample samples, oversampling samples and return samples in a sequence.
    """

    def __init__(self, x: XGenerator, y: Labels, name: str = ""):
        """

        :param x: The X part of an dataset. It contains a sequence of x and the corresponding index of the x.
        :param y: The y part of an dataset. It contains a sequence of y and the corresponding index of the y.
        :param name: Name of the dataset.
        """
        self.name = name
        self._x = x
        self._y = y
        mask_isin_x = np.isin(x.index, y.index)
        mask_isin_y = np.isin(y.index, x.index)
        self.index = self._y.index[mask_isin_y]
        assert np.array_equal(self.index, self._x.index[mask_isin_x])
        self.unique_classes, self.class_count = self.get_count_classes()
        self._class_indices = {}
        self.__range_index = np.arange(len(self.index), dtype=np.int)
        self.__range_index_x = np.arange(len(self._x.index), dtype=np.int)[mask_isin_x]
        self.__range_index_y = np.arange(len(self._y.index), dtype=np.int)[mask_isin_y]
        for i, class_ in enumerate(self.unique_classes):
            index_for_class = self._y.index[(self._y.labels == class_).all(1)]
            self._class_indices[str(i)] = self.__range_index[np.isin(self.index, index_for_class)]

        distri = self.class_count / np.sum(self.class_count)
        relation = np.max(distri) / np.min(distri)
        if relation > 4:
            msg = f"Classes highly unbalanced. Consider to change the labeling method or have a look. \n" \
                  f"N Highest class {self.unique_classes[np.argmax(distri)]}: {self.class_count[np.argmax(distri)]} \n" \
                  f"N Lowest class {self.unique_classes[np.argmin(distri)]}: {self.class_count[np.argmin(distri)]} \n" \
                  f"Ratio - n_class_max / n_class_min: {relation} \n"

            for class_, value in zip(self.unique_classes, self.class_count):
                msg = msg + "Class " + str(class_) + ": " + str(value) + "\n"
            logger.warning(msg)

    @property
    def x_shape(self) -> tuple:
        return self._x.shape

    @property
    def y_shape(self) -> tuple:
        return self._y.shape

    @property
    def with_index(self) -> bool:
        return self._y.with_index == self._x.with_index

    @with_index.setter
    def with_index(self, value: bool):
        assert (isinstance(value, bool))
        self._x.with_index = value
        self._y.with_index = value

    def __len__(self) -> int:
        return len(self.index)

    def get_unique_classes(self) -> np.array:
        unique_classes = np.unique(self._y.labels, axis=0)
        return unique_classes

    def get_count_classes(self) -> tuple:
        """
        Returns to lists. The first list is a list of all unique classes. The second list a the amount of samples per
        class. The index of the two lists can be used as the mapping.
        Example:
        unique_class_0 = unique_classes_list[0]
        unique_class_0_n_samples = class_count_list[0]
        :return:
        """
        unique_classes = np.unique(self._y.labels, axis=0)
        class_count = [
            len(self._y.labels[(self._y.labels == unique_class).all(1) & (np.isin(self._y.index, self.index))]) for
            unique_class in unique_classes]
        return unique_classes, class_count

    def get_shuffled_sample_from_specified_class(self, class_choice: np.ndarray):
        """
        Choice a class. This method returns a shuffled sample of this class.
        :param class_choice:
        :return:
        """
        if isinstance(class_choice, list):
            class_choice = np.asarray(class_choice)
        index_class = np.argmax((self.unique_classes == class_choice).all(1))
        possible_indices = self._class_indices[str(index_class)]
        index_choice_int = np.random.randint(0, len(possible_indices))
        index_choice = possible_indices[index_choice_int]
        return self[index_choice]

    def get_shuffled_sample(self):
        """
        Returns a shuffled sample.

        :return:
        """
        index_choice_int = np.random.randint(0, len(self.index))
        return self[index_choice_int]

    def get_oversampled_sample(self):
        """
        Returns a oversampled shuffled sample. The probability for each class is the same.

        :return:
        """
        class_choice_int = np.random.choice(range(len(self.unique_classes)),
                                            p=np.ones(len(self.unique_classes)) / len(self.unique_classes))
        class_choice = self.unique_classes[class_choice_int]
        return self.get_shuffled_sample_from_specified_class(class_choice)

    def _translate_int_index_to_int_x_y_range_index(self, index: int):
        int_index_x = self.__range_index_x[index]
        int_index_y = self.__range_index_y[index]
        return int_index_x, int_index_y

    @staticmethod
    def _translate_slice_to_range(slice_object: slice, target_array: np.ndarray) -> range:
        indices = slice_object.indices(len(target_array))
        indices_range = range(indices[0], indices[1], indices[2])
        return indices_range

    def __getitem__(self, key) -> np.array:
        if isinstance(key, Number):
            int_index_x, int_index_y = self._translate_int_index_to_int_x_y_range_index(key)
            return self._x[int_index_x], self._y[int_index_y]
        elif isinstance(key, datetime.datetime):
            return self._x[key], self._y[key]
        elif (isinstance(key, list) and isinstance(key[0], Number)) or isinstance(key, range):
            translated_indices_x = []
            translated_indices_y = []
            for index_ in key:
                x_index, y_index = self._translate_int_index_to_int_x_y_range_index(index_)
                translated_indices_x.append(x_index)
                translated_indices_y.append(y_index)
            return self._x[translated_indices_x], self._y[translated_indices_y]
        elif isinstance(key, slice):
            ranged_slice = self._translate_slice_to_range(key, self.index)
            return self[ranged_slice]

        else:
            msg = f"{type(key)} is not implemented."
            logger.erro(msg)
            raise NotImplementedError(msg)


class DataSetManager:
    """
    This Manager can efficiently handle multiple Datasets to train a ML Model. It contains a set of samples. Each Sample contains X, y and
    an index. This Dataset can efficiently random sample samples, oversampling samples from all datasets it manage.
    """

    def __init__(self, datasets: [Dataset]):
        self.datasets = dict()
        self.n_samples = 0
        self._proba_dataset_map = dict()
        for dataset in datasets:
            dataset.with_index = False
            self.add_dataset(dataset)

    @property
    def proba_dataset_map(self):
        return self._proba_dataset_map

    @property
    def x_shape(self):
        dataset_name = list(self.datasets.keys())[0]
        return self.datasets[dataset_name]["dataset"].x_shape

    @property
    def y_shape(self):
        dataset_name = list(self.datasets.keys())[0]
        return self.datasets[dataset_name]["dataset"].y_shape

    def add_dataset(self, other: Dataset):
        """
        This method can add another dataset into the datasetmanager.
        :param other: another instance of a Dataset
        :return:
        """
        dataset_dict = dict()
        dataset_dict["n_samples"] = len(other)
        dataset_dict["dataset"] = other
        dataset_name = other.name
        if dataset_name in self.datasets.keys():
            msg = f"A dataset with the same name {dataset_name} is already in this DataSetManager instance."
            logger.error(msg)
            raise KeyError(msg)
        self.datasets[other.name] = dataset_dict
        self._update_n_samples()
        self._update_proba_dataset()

    def _update_n_samples(self):
        n_samples = 0
        for name in self.datasets.keys():
            n_samples_single = self.datasets[name]["n_samples"]
            n_samples += n_samples_single
        self.n_samples = n_samples

    def _update_proba_dataset(self):
        proba_dataset_map = dict()
        for name in self.datasets.keys():
            n_samples_single = self.datasets[name]["n_samples"]
            proba_dataset_map[name] = n_samples_single / self.n_samples
        self._proba_dataset_map = proba_dataset_map

    def _get_dataset_with_prob(self) -> str:
        dataset_names = list(self._proba_dataset_map.keys())
        dataset_probs = list(self._proba_dataset_map.values())
        choice_dataset = np.random.choice(dataset_names,
                                          p=dataset_probs)
        return choice_dataset

    def get_shuffled_sample(self):
        """
        Returns a shuffled sample across all containing datasets.
        :return:
        """
        choice_dataset = self._get_dataset_with_prob()
        return self.datasets[choice_dataset]["dataset"].get_shuffled_sample()

    def get_oversampled_sample(self):
        """
        Returns a oversampled shuffled sample. The probability for each class is the same.

        :return:
        """
        choice_dataset = self._get_dataset_with_prob()
        return self.datasets[choice_dataset]["dataset"].get_oversampled_sample()


class BatchGenerator(keras.utils.Sequence):
    """
    This BatchGenerator can efficiently create batches from datasets. It can use Dataset classes or DataSetManager
    classes as a dataset.
    An instance of this class can be used in keras.model.fit_generator().
    It can generate random samples batches, batches from a sequence of samples in the right order and strided
    batches from a sequence of sample in the right order.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = False, oversampling: bool = False,
                 iterations: int = None, stride: int = None, x_only: bool = False):
        """

        :param dataset: A Dataset object.
        :param batch_size: The batch size of the returned batch
        :param shuffle: If the samples in a batch should be shuffled.
        :param oversampling: If the samples in a batch should be oversampled.
        :param iterations: The len of the iterator.
        :param stride: The stride of strided sequences.
        :param x_only: If the y batch should be returned
        """
        self.dataset = dataset
        if hasattr(dataset, "with_index"):
            self.dataset.with_index = False
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.oversampling = oversampling
        self.iterations = iterations
        self.stride = stride
        self.x_only = x_only
        self._x_shape = self.dataset.x_shape
        self._y_shape = self.dataset.y_shape
        self.current = 0
        if oversampling and not shuffle:
            msg = "Oversampling requires shuffling. Please decide on shuffle = False or oversampling = False."
            logger.error(msg)
            raise AssertionError(msg)
        if self.iterations and not shuffle:
            msg = "shuffle = False requires to iterate over the hole dataset. Please decide on shuffle = False or iterations."
            logger.error(msg)
            raise AssertionError(msg)
        if isinstance(self.dataset, DataSetManager) and (not self.shuffle):
            msg = f"The dataset is a type {DataSetManager.__name__}. This dataset must be shuffled."
            logger.error(msg)
            raise AssertionError(msg)

    @property
    def x_shape(self):
        return self._x_shape

    @property
    def y_shape(self):
        return self._y_shape

    @property
    def shape(self):
        return (self.x_shape, self.y_shape)

    def __getitem__(self, indices) -> np.array:
        if not self.stride:
            index_range = range(indices * self.batch_size, indices * self.batch_size + self.batch_size)
        else:
            index_range = range(indices * self.batch_size * self.stride,
                                indices * self.batch_size * self.stride + self.batch_size * self.stride, self.stride)

        batch_x = np.zeros((self.batch_size,) + self._x_shape[1:])
        batch_x.fill(np.nan)
        batch_y = np.empty((self.batch_size,) + self._y_shape[1:], dtype=np.int)

        for i, index in enumerate(index_range):
            if self.shuffle:
                batch_x[i], batch_y[i] = self.dataset.get_shuffled_sample()
            elif self.oversampling:
                batch_x[i], batch_y[i] = self.dataset.get_oversampled_sample()
            else:
                batch_x[i], batch_y[i] = self.dataset[index]
        if not self.x_only:
            return batch_x, batch_y
        else:
            return batch_x

    def __len__(self) -> int:
        if self.iterations:
            return self.iterations
        if self.stride:
            return len(self.index) // self.batch_size
        else:
            return len(self.dataset) // self.batch_size

    @property
    def index(self) -> np.array:
        if self.shuffle:
            logger.info("Shuffle = True. No index can be returned")
        elif not self.shuffle and not self.iterations and not self.oversampling and not self.stride:
            return self.dataset.index, self.dataset.index
        elif not self.shuffle and not self.iterations and not self.oversampling and self.stride:
            return self.dataset.index[::self.stride][
                   :len(self.dataset.index[::self.stride]) // self.batch_size * self.batch_size]
        else:
            msg = "Not implemented"
            logger.error(msg)
            raise NotImplementedError(msg)

    def on_epoch_end(self):
        self.current = 0

    def __next__(self):
        """ Calls the next item."""
        if self.current < len(self):
            item = self[self.current]
            self.current += 1
            return item
        else:
            raise StopIteration

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item


def transform_to_rolling_windows(index, x, y, window):
    index_transformed = index[window - 1:index.shape[0]]
    y_transformed = y[window - 1:index.shape[0]]
    shape = x.shape[0] - window + 1, window, x.shape[-1]
    strides = (x.strides[0], x.strides[0], x.strides[1])
    windowed_x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return index_transformed, windowed_x, y_transformed


def transform_to_windowed(x, window):
    shape = (x.shape[0] - window + 1, window) + x.shape[1:]
    strides = (x.strides[0], x.strides[0]) + x.strides[1:]
    windowed_x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return windowed_x
