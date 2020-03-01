import numpy as np
import time
from src.datagenerator import Labels, XGenerator, Dataset, BatchGenerator
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten

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

mock_labels = np.repeat(np.arange(len_mock, dtype=np.int), 2).reshape((len_mock, 2))

x_exist = np.ones(mock_index.shape, dtype=np.bool)
x_non_exist = 30
x_exist[:x_non_exist] = False

if __name__ == "__main__":
    labels = Labels(labels=mock_labels, index=mock_index, with_index=False)
    x_generator = XGenerator(raw_sequence=mock_sequence,
                             raw_index=mock_index, x_window=x_window, x_exist=x_exist,
                             with_index=False)
    dataset = Dataset(x=x_generator, y=labels, name="tutorial")

    batch_size = 10
    batch_generator = BatchGenerator(dataset=dataset, batch_size=batch_size, shuffle=True, stride=None)

    samples = len(batch_generator)
    start = time.clock()

    test_batch = batch_generator[0]
    duration = time.clock() - start
    print(f"Samples per second:{samples * batch_size / duration}")
    print(test_batch[0].shape, test_batch[1].shape)
    print(f"Batches per second:{samples / duration}")

    a = Input(shape=(batch_generator.x_shape[1:]))
    b = Flatten()(a)
    c = Dense(units=2, activation="softmax")(b)
    model = KerasModel(inputs=a, outputs=c)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy")
    model.summary()
    start = time.clock()
    model.fit_generator(batch_generator, shuffle=False, workers=1)
    duration = time.clock() - start
    print(f"Trainings duration: {duration}")

    start = time.clock()
    batch_generator = BatchGenerator(dataset=dataset, batch_size=10, shuffle=False, stride=None, x_only=True)
    prediction = model.predict_generator(batch_generator)
    duration = time.clock() - start
    print(f"Trainings duration: {duration}")

    labels = Labels(labels=mock_labels, index=mock_index, with_index=False)
    x_generator = XGenerator(raw_sequence=mock_sequence,
                             raw_index=mock_index, x_window=x_window, x_exist=x_exist,
                             with_index=False)
    dataset = Dataset(x=x_generator, y=labels, name="tutorial")
    batch_generator = BatchGenerator(dataset=dataset, batch_size=20, shuffle=False, stride=10, x_only=False)
    for batch in batch_generator:
        print(batch[1])
