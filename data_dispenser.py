from pathlib import Path
import utils as ut
import tensorflow as tf
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os

class CelebADispenser():

    def __init__(self, data_dir=Path('data'), screen_size=64):
        self.data_dir = data_dir
        self.height, self.width = screen_size, screen_size
        self.prepare_data()

    def get_random_target(self, num=1, squeeze=False, train=True):
        data_length = self.real_data.shape[0]
        train_len = int(data_length*0.9)
        if train:
          a = train_len
        else:
          a = np.arange(train_len, data_length)
        random_idxes = np.random.choice(a, num, replace=False)
        random_image = self.real_data[random_idxes]
        if squeeze:
            random_image = np.squeeze(random_image, 0)
        return random_image
      
    def prepare_data(self):
        ut.io.makedirs(self.data_dir)
        
        omniglot_image_files = tf.gfile.Glob('data/img_align_celeba/*')

        # ground truth omniglot data
        mnist_dir = self.data_dir / 'celeba_full'
        try:
          os.makedirs(str(mnist_dir))
        except OSError:
          pass
        pkl_path = mnist_dir / 'celeba_dict.pkl'
        if pkl_path.exists():
            self.real_data = np.load(pkl_path)
        else:
            omniglot_list = []
            #iterator = tqdm(omniglot_image_files, desc="Processing")
            for idx, img in enumerate(omniglot_image_files):
              if idx % 10000 == 0: print(idx)
              img = ut.io.imread(img)
              img = ut.io.imresize(
                                img, [self.height, self.width],
                                interp='cubic')
              omniglot_list.append(img)
                          
            self.real_data = np.concatenate(omniglot_list).reshape(-1, 64, 64, 3)
            with open(str(pkl_path), 'w') as f:
              np.save(f, self.real_data)
                        
            
class MNISTDispenser():
    def __init__(self, data_dir=Path('data'), screen_size=64):
        self.data_dir = data_dir
        self.height, self.width = screen_size, screen_size
        self.prepare_mnist()

#     def get_random_target(self, num=1, squeeze=False, train=True):
#         random_idxes = np.random.choice(self.real_data.shape[0], num, replace=False)
#         random_image = self.real_data[random_idxes]
#         if squeeze:
#             random_image = np.squeeze(random_image, 0)
#         return random_image

    def get_random_target(self, num=1, squeeze=False, train=True):
        data_length = self.real_data.shape[0]
        train_len = int(data_length*0.9)
        if train:
          a = train_len
        else:
          a = np.arange(train_len, data_length)
        random_idxes = np.random.choice(a, num, replace=False)
        random_image = self.real_data[random_idxes]
        if squeeze:
            random_image = np.squeeze(random_image, 0)
        return random_image
      

    def prepare_mnist(self):
        ut.io.makedirs(self.data_dir)

        # ground truth MNIST data
        mnist_dir = self.data_dir / 'mnist'
        pkl_path = mnist_dir / 'mnist_dict.pkl'

        if pkl_path.exists():
            mnist_dict = ut.io.load_pickle(pkl_path)
        else:
            mnist = tf.contrib.learn.datasets.DATASETS['mnist'](str(mnist_dir))
            mnist_dict = defaultdict(lambda: defaultdict(list))
            for name in ['train', 'test', 'valid']:
                for num in range(10):
                    filtered_data = \
                            mnist.train.images[mnist.train.labels == num]
                    filtered_data = \
                            np.reshape(filtered_data, [-1, 28, 28])

                    iterator = tqdm(filtered_data,
                                    desc="[{}] Processing {}".format(name, num))
                    for idx, image in enumerate(iterator):
                        # XXX: don't know which way would be the best
                        resized_image = ut.io.imresize(
                                image, [self.height, self.width],
                                interp='cubic')
                        mnist_dict[name][num].append(
                                np.expand_dims(resized_image, -1))
            ut.io.dump_pickle(pkl_path, mnist_dict)

        mnist_dict = mnist_dict['train'] # as opposed to test

        data = []
        for num in range(10):
            data.append(mnist_dict[int(num)])

        self.real_data = 255 - np.concatenate([d for d in data])
#         self.real_data_labels = np.sort(mnist.train.labels)
        
def celeba():
    celeba_dispenser = CelebADispenser()
    print(celeba_dispenser.get_random_target().shape)

def mnist():
    mnist_dispenser = MNISTDispenser()
    print(mnist_dispenser.get_random_target().shape)

if __name__=="__main__":
#     mnist()
    celeba()