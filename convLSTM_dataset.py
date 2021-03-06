from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

from skimage import io, transform
from torchvision import transforms, utils

import numpy as np
import os
import glob

from torch.utils.data import DataLoader


#  The pytorch dataset obj for binary target dataset
class convLSTM_Dataset(Dataset):
    def __init__(self, dataset_dir, n_class, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with force torque values and position of pressing.
                init_img_path (string): path to the  init image
                imgs_path (string): path to the image when pressing
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = dataset_dir
        self.n_feature = n_class

        self.transform = transform

        self.class_0_dir = os.path.join(self.dataset_dir, '0')
        self.class_1_dir = os.path.join(self.dataset_dir, '1')

        self.class_0_files = glob.glob(os.path.join(self.class_0_dir, '*'))
        self.class_1_files = glob.glob(os.path.join(self.class_1_dir, '*'))

        self.Nx, self.Ny = 30, 30

    def __len__(self):
        return len(self.class_0_files) + len(self.class_1_files)

    def __getitem__(self, idx):
        feature_size = self.Nx * self.Ny
        if idx < 0.5*self.__len__():
            target = 0
            frames = pd.read_csv(self.class_0_files[idx])
            # data = frames.values
        else:
            target = 1
            # print int(idx-0.5*self.__len__())
            frames = pd.read_csv(self.class_1_files[int(idx-0.5*self.__len__())])

        data = frames.values

        # print 'data size'+str(data.shape)

        # form a tensor (T, c, H, W)
        frame_matrix = np.zeros((data.shape[1], 3, self.Nx, self.Ny))
        for i in range(0, data.shape[1]):
            dx_interp = data[:feature_size, i]
            dy_interp = data[feature_size:, i]
            #         print dx_interp.shape, dy_interp.shape

            mag = np.sqrt(dx_interp ** 2 + dy_interp ** 2)

            dx_resized = dx_interp.reshape(self.Nx, self.Ny)
            dy_resized = dy_interp.reshape(self.Nx, self.Ny)
            mag_resized = mag.reshape(self.Nx, self.Ny)

            # stack them along axis 0
            temp = [dx_resized, dy_resized, mag_resized]
            matrix_3 = np.stack(temp, axis=0)

            frame_matrix[i,:,:,:] = matrix_3

        frame_matrix = np.flip(frame_matrix, axis=0)

        # print frame_matrix.shape




        # imgs_name = os.path.join(self.imgs_dir, self.csv_handler.iloc[idx, 0])
        # init_img_name = self.init_img_path
        #
        # imgs = io.imread(imgs_name, plugin='matplotlib')
        # init_img = io.imread(init_img_name, plugin='matplotlib')
        #
        # targets = self.csv_handler.iloc[idx, 1:self.n_feature+1].as_matrix()
        # print(targets)
        # targets = targets.astype('float').reshape(-1, 2)
        sample = {'frames': frame_matrix, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class convLSTM_tdiff_Dataset(Dataset):
    def __init__(self, dataset_dir, n_class, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with force torque values and position of pressing.
                init_img_path (string): path to the  init image
                imgs_path (string): path to the image when pressing
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = dataset_dir
        self.n_feature = n_class

        self.transform = transform

        self.class_0_dir = os.path.join(self.dataset_dir, '0')
        self.class_1_dir = os.path.join(self.dataset_dir, '1')

        self.class_0_files = glob.glob(os.path.join(self.class_0_dir, '*'))
        self.class_1_files = glob.glob(os.path.join(self.class_1_dir, '*'))

        self.Nx, self.Ny = 30, 30

    def __len__(self):
        return len(self.class_0_files) + len(self.class_1_files)

    def __getitem__(self, idx):
        feature_size = self.Nx * self.Ny
        if idx < 0.5*self.__len__():
            target = 0
            frames = pd.read_csv(self.class_0_files[idx])
            # data = frames.values
        else:
            target = 1
            frames = pd.read_csv(self.class_1_files[idx-0.5*self.__len__()])

        data = frames.values

        # print 'data size'+str(data.shape)

        # form a tensor (T, c, H, W)
        frame_matrix = np.zeros((data.shape[1]-1, 3, self.Nx, self.Ny))
        for i in range(1, data.shape[1]):
            dx = data[:feature_size, i]
            dy = data[feature_size:, i]
            # print dx_interp.shape, dy_interp.shape
            mag = np.sqrt(dx ** 2 + dy ** 2)
            # ---------------------------------
            dx_last = data[:feature_size, i - 1]
            dy_last = data[feature_size:, i - 1]
            mag_last = np.sqrt(dx_last**2 + dy_last**2)


            dx_diff = dx - dx_last
            dy_diff = dy - dy_last
            mag_diff = mag - mag_last


            dx_resized = dx_diff.reshape(self.Nx, self.Ny)
            dy_resized = dy_diff.reshape(self.Nx, self.Ny)
            mag_resized = mag_diff.reshape(self.Nx, self.Ny)

            # stack them along axis 0
            temp = [dx_resized, dy_resized, mag_resized]
            matrix_3 = np.stack(temp, axis=0)

            frame_matrix[i-1,:,:,:] = matrix_3

        frame_matrix = np.flip(frame_matrix, axis=0)

        # print frame_matrix.shape




        # imgs_name = os.path.join(self.imgs_dir, self.csv_handler.iloc[idx, 0])
        # init_img_name = self.init_img_path
        #
        # imgs = io.imread(imgs_name, plugin='matplotlib')
        # init_img = io.imread(init_img_name, plugin='matplotlib')
        #
        # targets = self.csv_handler.iloc[idx, 1:self.n_feature+1].as_matrix()
        # print(targets)
        # targets = targets.astype('float').reshape(-1, 2)
        sample = {'frames': frame_matrix, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

import IPython


class convLSTM_Dataset_dxdy(Dataset):
    def __init__(self, dataset_dir, n_class, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with force torque values and position of pressing.
                init_img_path (string): path to the  init image
                imgs_path (string): path to the image when pressing
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = dataset_dir
        self.n_feature = n_class

        self.transform = transform

        self.class_files = {}
        for cl in range(0, n_class):
            self.class_files.update({ str(cl): glob.glob(os.path.join(self.dataset_dir, str(cl), '*')) })
        # IPython.embed()

        self.Nx, self.Ny = 30, 30

        self.length_cummu = []
        self.length_cummu.append(len(self.class_files[str(0)]))
        for cl in range(1, self.n_feature):
            self.length_cummu.append(len(self.class_files[str(0)]) + self.length_cummu[cl-1])
        # print self.length_cummu


    def __len__(self):

        return self.length_cummu[-1]

    def __getitem__(self, idx):
        # check if the idx is valid
        # print self.length_cummu[-1]
        if idx < 0:
            print('Please provide positive integers.')
            return
        if idx >= self.length_cummu[-1]:
            print('The retrieving index is over the size of dataset.')
            return

        feature_size = self.Nx * self.Ny

        for cl in range(0, self.n_feature):
            if cl == 0:
                last_length = 0
            else:
                last_length = self.length_cummu[cl-1]

            if idx < self.length_cummu[cl]:
                target = int(cl)
                # print self.class_files[str(cl)][int(idx - last_length)]
                frames = pd.read_csv(self.class_files[str(cl)][int(idx - last_length)])
                break

        data = frames.values

        # print 'data size'+str(data.shape)

        # form a tensor (T, c, H, W)
        frame_matrix = np.zeros((data.shape[1], 2, self.Nx, self.Ny))
        for i in range(0, data.shape[1]):
            dx_interp = data[:feature_size, i]
            dy_interp = data[feature_size:, i]
            #         print dx_interp.shape, dy_interp.shape

            dx_resized = dx_interp.reshape(self.Nx, self.Ny)
            dy_resized = dy_interp.reshape(self.Nx, self.Ny)

            # stack them along axis 0
            frame_matrix[i,:,:,:] = np.stack([dx_resized, dy_resized], axis=0)

        # frame_matrix = np.flip(frame_matrix, axis=0)

        sample = {'frames': frame_matrix, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class LSTM_Dataset(Dataset):
    def __init__(self, dataset_dir, n_class, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with force torque values and position of pressing.
                init_img_path (string): path to the  init image
                imgs_path (string): path to the image when pressing
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = dataset_dir
        self.n_feature = n_class

        self.transform = transform

        self.class_files = {}
        for cl in range(0, n_class):
            self.class_files.update({str(cl): glob.glob(os.path.join(self.dataset_dir, str(cl), '*'))})
        # IPython.embed()

        self.Nx, self.Ny = 30, 30

        self.length_cummu = []
        self.length_cummu.append(len(self.class_files[str(0)]))
        for cl in range(1, self.n_feature):
            self.length_cummu.append(len(self.class_files[str(0)]) + self.length_cummu[cl-1])
        # print self.length_cummu

    def __len__(self):

        return self.length_cummu[-1]

    def __getitem__(self, idx):
        # check if the idx is valid
        # print self.length_cummu[-1]
        if idx < 0:
            print('Please provide positive integers.')
            return
        if idx >= self.length_cummu[-1]:
            print('The retrieving index is over the size of dataset.')
            return

        # feature_size = self.Nx * self.Ny

        for cl in range(0, self.n_feature):
            if cl == 0:
                last_length = 0
            else:
                last_length = self.length_cummu[cl-1]

            if idx < self.length_cummu[cl]:
                target = int(cl)
                # print self.class_files[str(cl)][int(idx - last_length)]
                frames = pd.read_csv(self.class_files[str(cl)][int(idx - last_length)])
                break

        data = frames.values

        # print 'data size'+str(data.shape)

        # form a tensor (T, 2*feature_size) here the shape should be (15, 1800)
        frame_matrix = data.transpose()

        sample = {'frames': frame_matrix, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomHorizontalFlip(object):
    """Randomly flip the numpy array(as image) horizontally
    This transform should be put before ToTensor()

        input: numpy array (/or array sequence) with size of (Sequence, channel, height, width)
        output: numpy array with same size (flipped or not depends on random number)
    Args:
        p: possibility of flip

    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image_sequence, target = sample['frames'], sample['target']

        if np.random.uniform() > self.p:
            image_sequence = np.flip(image_sequence, axis=3)

        return {'frames': image_sequence, 'target': target}


class RandomVerticalFlip(object):
    """Randomly flip the numpy array(as image) vertically
    This transform should be put before ToTensor()

        input: numpy array (/or array sequence) with size of (Sequence, channel, height, width)
        output: numpy array with same size (flipped or not depends on random number)
    Args:
        p: possibility of flip

    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image_sequence, target = sample['frames'], sample['target']

        if np.random.uniform() > self.p:
            image_sequence = np.flip(image_sequence, axis=2)

        return {'frames': image_sequence, 'target': target}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, init_image, targets = sample['image'], sample['init_image'], sample['targets']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        img_init = transform.resize(init_image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'init_image': img_init, 'targets': targets}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        frames, target = sample['frames'], sample['target']
        target = np.array(target)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        # targets = target.astype(float)  # origin format too long to use from_numpy

        # image = image.transpose((2, 0, 1))
        # init_image = init_image.transpose((2,0,1))
        return {'frames': torch.from_numpy(frames.copy()),
                'target': torch.from_numpy(target)}


if __name__ == '__main__':
    convlstm_dataset = convLSTM_Dataset_dxdy(dataset_dir='../dataset/resample_skipping_stride1',
                                                n_class=4,
                                                transform=transforms.Compose([
                                                            RandomHorizontalFlip(),
                                                            RandomVerticalFlip(),
                                                            ToTensor()])
                                            )

    conv = convlstm_dataset[100]['frames'].shape
    print conv
    print len(convlstm_dataset)

    lstm_dataset = LSTM_Dataset(dataset_dir='../dataset/resample_skipping_stride1',
                                n_class=4,
                                transform=transforms.Compose([
                                                            ToTensor()])
                                )

    lstm = lstm_dataset[100]['frames'].shape
    print lstm, len(lstm_dataset)


    # print conv['frames']
    #
    #
    #
    # dataloader = DataLoader(convlstm_dataset, batch_size=8, shuffle=True, num_workers=4)
    # print(len(convlstm_dataset))

    # sample = convlstm_dataset[100]
    # print sample['frames']

    #=====================================
    # convlstm_tdiff_dataset = convLSTM_tdiff_Dataset(dataset_dir='../dataset/resample_skipping',
    #                                                 n_class=2,
    #                                                 transform=transforms.Compose([
    #                                                     RandomHorizontalFlip(),
    #                                                     RandomVerticalFlip(),
    #                                                     ToTensor()])
    #                                                 )
    # dataloader_tdiff = DataLoader(convlstm_tdiff_dataset, batch_size=8, shuffle=True, num_workers=4)
    # sample_tdiff = convlstm_tdiff_dataset[100]
    # print sample_tdiff['frames'].shape



    # fig = plt.figure()

    # for i in range(len(convlstm_dataset)):
    #     sample = convlstm_dataset[i]
    #
    #     # print(i, sample['image'].size(), sample['init_image'].size(), sample['targets'].size())
    #     print(i, sample['frames'].size(), sample['target'])


    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['frames'].size(),
    #           sample_batched['target'])


