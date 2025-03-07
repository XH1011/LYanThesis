import numpy as np
import pickle

class DataProvider(object):
    def __init__(self, images, labels, reshape=True):
        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        if reshape:
            #assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate(
                (images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

def read_data_sets(path_in):
    with open(path_in,'rb') as f:
        images_train, labels_train, images_test, labels_test = pickle.load(f)
    print('for train:',images_train.shape)
    print('for test:',images_test.shape)
    return DataProvider(images_train, labels_train), \
           DataProvider(images_test, labels_test)

def read_features(path_in):
    with open(path_in,'rb') as f:
        images_train, images_test = pickle.load(f)
    media = images_train.mean(axis=0)
    des = images_train.std(axis=0)
    images_train = (images_train - media) / des
    images_test = (images_test - media) / des
    images_train = images_train.reshape((-1,1,images_train.shape[1]))
    images_test = images_test.reshape((-1, 1, images_test.shape[1]))
    print('for train:',images_train.shape)
    print('for test:',images_test.shape)
    return DataProvider(images_train, np.zeros((images_train.shape[0]))), \
           DataProvider(images_test, np.zeros((images_test.shape[0])))