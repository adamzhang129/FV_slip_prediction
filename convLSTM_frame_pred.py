import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, n_frames_ahead):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.n_frames_ahead = n_frames_ahead
        self.hidden_size = hidden_size
        self.Gates_layer1 = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

        self.Gates_layer2 = nn.Conv2d(2*hidden_size, 4*hidden_size, KERNEL_SIZE, padding=PADDING)

        self.height, self.width = 30, 30

        self.Shrink = nn.Conv2d(hidden_size, self.input_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # print spatial_size

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size1 = [batch_size, self.hidden_size] + list(spatial_size)  # (B,Hidden_size, H, W)
            prev_state1 = (
                Variable(torch.zeros(state_size1)).type(torch.cuda.FloatTensor),
                Variable(torch.zeros(state_size1)).type(torch.cuda.FloatTensor)
            )  # list of h[t-1] and C[t-1]: both of size [batch_size, hidden_size, D, D]

            # =======layer2 lstm previous states
            state_size2 = [batch_size, self.hidden_size] + list(spatial_size)

            prev_state2 = (
                Variable(torch.zeros(state_size2)).type(torch.cuda.FloatTensor),
                Variable(torch.zeros(state_size2)).type(torch.cuda.FloatTensor)
            )  # list of h[t-1] and C[t-1]: both of size [batch_size, hidden_size, D, D]

            prev_state = (prev_state1, prev_state2)

        prev_state1, prev_state2 = prev_state
        prev_hidden1, prev_cell1 = prev_state1
        prev_hidden2, prev_cell2 = prev_state2

        # data size is [batch, channel, height, width]
        # print input_.type(), prev_hidden.type()
        stacked_inputs = torch.cat((input_, prev_hidden1), 1)  # concat x[t] with h[t-1]
        gates = self.Gates_layer1(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # print cell_gate.shape

        # compute current cell and hidden state
        cell1 = (remember_gate * prev_cell1) + (in_gate * cell_gate)
        hidden1 = out_gate * f.tanh(cell1)

        # print hidden.size()
        # conv2 = self.Conv(hidden)
        # flat = conv2.view(-1, conv2.size(1) * conv2.size(2) * conv2.size(3))

        # =============layer 2 gates operation =================================
        # feed hidden state from layer 1 to layer 2 as input
        stacked_inputs2 = torch.cat((hidden1, prev_hidden2), 1)  # concat x[t] with h[t-1]
        gates2 = self.Gates_layer2(stacked_inputs2)

        # chunk across channel dimension
        in_gate2, remember_gate2, out_gate2, cell_gate2 = gates2.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate2 = torch.sigmoid(in_gate2)
        remember_gate2 = torch.sigmoid(remember_gate2)
        out_gate2 = torch.sigmoid(out_gate2)

        # apply tanh non linearity
        cell_gate2 = torch.tanh(cell_gate2)
        # print cell_gate.shape

        # compute current cell and hidden state
        cell2 = (remember_gate2 * prev_cell2) + (in_gate2 * cell_gate2)
        hidden2 = out_gate2 * f.tanh(cell2)

        # flat = hidden2.view(-1, hidden2.size(1)*hidden2.size(2)*hidden2.size(3))
        # # print flat.size()
        # out = self.linear(flat)
        # out = self.dropout(out)

        out = self.Shrink(hidden2)

        # print out.shape

        # print out.shape
        # out = out.view(-1, self.n_frames_ahead, self.input_size, self.height, self.width)
        # print out.shape
        # out = torch.transpose(out, 0, 1)

        return out, ((hidden1, cell1), (hidden2, cell2))



from convLSTM_dataset import *
from torch.utils.data import DataLoader

from torch.utils.data.dataset import random_split

# from logger import Logger
# logger = Logger('./logs')

import IPython

from torch.utils.data.sampler import SubsetRandomSampler


def random_split_customized(dataset, train_ratio=0.9, shuffle_dataset=True):
    random_seed = 41
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    # print indices
    split = int(np.floor(train_ratio * dataset_size))
    # print split
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # print indices[0:10]
    train_indices, test_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, test_sampler


from skimage import data, img_as_float
from skimage.measure import compare_ssim


def image_similarity_metrics(img1, img2):

    # images are with shape of (batch_size, channel, height, width)
    # ========MSE===================
    shape = img1.shape
    if len(shape) == 4:
        B = shape[0]  # batch_size
    numel = shape[-3] * shape[-2] * shape[-1]

    output_matrix = []
    for ind_B in range(0, B):
        im1 = img1[ind_B]
        im2 = img2[ind_B]
        im1 = np.moveaxis(im1, 0, -1)
        im2 = np.moveaxis(im2, 0, -1)

        # IPython.embed()
        l2 = np.linalg.norm((im1 - im2)) ** 2 / numel
        # ========L1====================
        l1 = np.sum(np.abs(im1 - im2)) / numel
        # =======SSIM===================
        ssim = compare_ssim(im1, im2, data_range=img2.max() - img2.min(), multichannel=True)

        output_matrix.append([l1, l2, ssim])

    return np.array(output_matrix)

def generate_dataloader(path, batch_size):
    convlstm_dataset = convLSTM_Dataset(dataset_dir=path,
                                        n_class=2,
                                        transform=transforms.Compose([
                                            RandomHorizontalFlip(),
                                            RandomVerticalFlip(),
                                            ToTensor()])
                                        )

    train_sampler, _ = random_split_customized(convlstm_dataset, train_ratio=0.9)

    train_dataloader = DataLoader(convlstm_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=4)

    # test set loaded without augmentation
    convlstm_dataset_wo_flip = convLSTM_Dataset(dataset_dir=path,
                                        n_class=2,
                                        transform=transforms.Compose([
                                            ToTensor()])
                                        )

    _, test_sampler = random_split_customized(convlstm_dataset_wo_flip, train_ratio=0.9)
    test_dataloader = DataLoader(convlstm_dataset, batch_size=batch_size, sampler=test_sampler,
                                 num_workers=4)

    return train_sampler, test_sampler, train_dataloader, test_dataloader

def construct_metrics_table(values, metrics=['L1', 'L2', 'SSIM']):
    N = len(values)
    n_frames_ahead = range(1, N+1)
    index = list(map(str, n_frames_ahead))
    array1 = np.repeat(n_frames_ahead, 3)
    # array1 = np.concatenate((array1, np.repeat(['Ave'], 3)), axis=0)
    array2 = np.tile(metrics, N)
    # IPython.embed()

    arrays = np.stack((array1, array2))

    tuples = list(zip(*arrays))
    column = pd.MultiIndex.from_tuples(tuples, names=['Future Frame Number', 'Metrics'])

    df = pd.DataFrame(values, index=index, columns=column)
    df.to_csv('metric results.csv')

import time

def _main():
    """
    Run some basic tests on the API
    """
    # define batch_size, channels, height, width
    batch_size, channels, height, width = 32, 3, 30, 30
    hidden_size = 32 # 64           # hidden state size
    lr = 1e-5     # learning rate
    max_epoch = 1  # number of epochs

    dataset_path = '../dataset/resample_skipping'
    train_sampler, test_sampler, train_dataloader, test_dataloader = generate_dataloader(dataset_path, batch_size)

    train_loss_cache = []
    test_loss_cache = []

    max_frames_ahead = 5
    n_metrics = 3
    metric_table = np.zeros([max_frames_ahead, max_frames_ahead*n_metrics]).astype(str)
    # train with different values of n_frames_ahead to see the performance
    for n_frames_ahead in range(1, 6):
        n_frames = 10 - n_frames_ahead

        print '\n =============[Train with n_frames_ahead = {} ================]'.format(n_frames_ahead)
        print('Instantiate model')
        model = ConvLSTMCell(channels, hidden_size, n_frames_ahead)
        print(repr(model))

        if torch.cuda.is_available():
            # print 'sending model to GPU'
            model = model.cuda()

        print('Create input and target Variables')
        x = Variable(torch.rand(n_frames, batch_size, channels, height, width))
        # y = Variable(torch.randn(T, b, d, h, w))
        y = Variable(torch.rand(batch_size))

        print('Create a MSE criterion')
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)

        model.train()

        # -----------------------------------------------------------
        print('Start the training, Running for', max_epoch, 'epochs')
        for epoch in range(0, max_epoch):
            loss_train = 0
            n_right_train = 0

            for step, sample_batched in enumerate(train_dataloader):
                model = model.train()
                loss = 0

                frames = sample_batched['frames']

                # y = sample_batched['target']
                # transpose time sequence and batch (sequence, batch, channel, height, width)
                frames = torch.transpose(frames, 0, 1)

                x = frames[:n_frames]
                y = frames[n_frames:]
                # IPython.embed()
                # x = x.type(torch.FloatTensor)
                # print x.size()

                if torch.cuda.is_available():
                    # print 'sending input and target to GPU'
                    x = x.type(torch.cuda.FloatTensor)
                    y = y.type(torch.cuda.FloatTensor)

                state = None
                out = None

                # IPython.embed()
                for t in range(0, n_frames):
                    # print x[t,0,0,:,:]
                    out, state = model(x[t], state)
                    if t in range(0, n_frames)[-n_frames_ahead:]:
                        # IPython.embed()
                        loss += loss_fn(out, y[n_frames_ahead - (n_frames - t)])

                # reduce loss to be loss on single frame discrepancy/loss

                # zero grad parameters
                model.zero_grad()

                # compute new grad parameters through time!
                loss.backward()
                optimizer.step()

                loss_train += loss.item() * batch_size / n_frames_ahead


                Step = 20
                if (step + 1) % Step == 0:
                    loss_train_reduced = loss_train / (Step * batch_size)
                    loss_train = 0.
                    print '         =================================================================='
                    print ('        [TRAIN set] Epoch {}, Step {}, Average Loss (every 20 steps): {:.6f}'
                           .format(epoch, step + 1, loss_train_reduced))


        # model_path = './saved_model/convlstm_frame_predict_20190311_200epochs_3200data_flipped_{}f_ahead.pth'\
        #     .format(n_frames_ahead)
        # torch.save(model.state_dict(), model_path)

        train_loss_cache.append(loss_train_reduced)


        print '-----Starting the evaluation over the test set.....'
        model = model.eval()


        n_metrics = 3
        metric_tmp = np.zeros((len(test_sampler), n_frames_ahead*n_metrics))
        start = time.time()
        loss_test = 0
        for test_step, test_sample_batched in enumerate(test_dataloader):
            loss = 0.

            frames = test_sample_batched['frames']
            # y = test_sample_batched['target']
            frames = torch.transpose(frames, 0, 1)
            # x = x.type(torch.FloatTensor)
            x = frames[:n_frames]
            y = frames[n_frames:]
            # IPython.embed()

            if torch.cuda.is_available():
                # print 'sending input and target to GPU'
                x = x.type(torch.cuda.FloatTensor)
                y = y.type(torch.cuda.FloatTensor)

            state_test = None
            out_test = None


            for t in range(0, n_frames):
                out_test, state_test = model(x[t], state_test)
                if t in range(0, n_frames)[-n_frames_ahead:]:
                    # IPython.embed()
                    loss += loss_fn(out_test, y[n_frames_ahead - (n_frames - t)])
                    # calculate different metrics
                    n = n_frames_ahead - (n_frames - t)
                    metric = image_similarity_metrics(out_test.cpu().detach().numpy(), y[n].cpu().detach().numpy())
                    metric_tmp[test_step*batch_size:(test_step+1)*batch_size, 3*n:3*(n+1)] = metric

            loss_test += loss.item() * batch_size / n_frames_ahead

        # ---------------------------------

        # print metric_tmp.shape
        mu = np.mean(metric_tmp, axis=0)
        sigma = np.var(metric_tmp, axis=0)
        print mu, sigma
        # IPython.embed()
        for n in range(0, len(mu)):
            metric_table[n_frames_ahead-1, n] = '{:06.4f}+/-{:06.4f}'.format(mu[n], sigma[n])

        loss_test_reduced = loss_test / len(test_sampler)
        print ('-----[TEST set] Average MSELoss (over all set): {:.6f}'
               .format(loss_test_reduced))

        gt = y.squeeze()[1][0]
        gt = gt.cpu().detach().numpy()
        out_single = out_test[0].cpu().detach().numpy()

        test_loss_cache.append(loss_test_reduced)

        # IPython.embed()
        # show_two_img(gt, out_single)

    #=======save metric results to file===========
    construct_metrics_table(metric_table)

    # random output a comparison of images
    image_prediction_comparison(n_frames_ahead=3, dataloader=test_dataloader)
    IPython.embed()



# import cv2
#
# def vec_color_encoding(x, y, encoding='hsv'):
#     if not x.shape == y.shape:
#         print '2d vector components should have same shapes.'
#         return None
#     hsv = np.zeros((x.shape[0], x.shape[1], 3))
#     hsv[..., 1] = 255
#
#     mag, ang = cv2.cartToPolar(x, y)
#
#     hsv[...,0] = ang*180/np.pi/2
#     hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#
#     hsv = np.uint8(hsv)
#     if encoding == 'hsv':
#         return hsv
#     elif encoding == 'rgb':
#         bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
#         return bgr


def image_prediction_comparison(n_frames_ahead, dataloader):
    n_frames = 10-n_frames_ahead
    batch_size, channels, height, width = 32, 3, 30, 30
    hidden_size = 32
    print('Instantiating model...')
    model = ConvLSTMCell(channels, hidden_size, n_frames_ahead)
    print(repr(model))

    model_path = './saved_model/convlstm_frame_predict_20190311_200epochs_3200data_flipped_{}f_ahead.pth'.format(n_frames_ahead)

    model.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        model = model.cuda()

    for test_step, test_sample_batched in enumerate(dataloader):
        loss = 0.

        frames = test_sample_batched['frames']
        # y = test_sample_batched['target']
        frames = torch.transpose(frames, 0, 1)
        # x = x.type(torch.FloatTensor)
        x = frames[:n_frames]
        y = frames[n_frames:]

        if torch.cuda.is_available():
            # print 'sending input and target to GPU'
            x = x.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)

        state_test = None
        out_test = None

        image_hat = []
        for t in range(0, n_frames):
            out_test, state_test = model(x[t], state_test)
            if t in range(0, n_frames)[-n_frames_ahead:]:
                image_hat.append(out_test)

        # random sample from this mini batch
        np.random.seed(10)
        i = np.random.randint(0, batch_size)

        fig, axes = plt.subplots(nrows=2, ncols=n_frames_ahead, figsize=(10, 4),
                                 sharex=True, sharey=True)
        # ax = axes.ravel()

        for n in range(0, n_frames_ahead):
            # IPython.embed()
            img_hat = image_hat[n][i].cpu().detach().numpy()
            img = y[n, i].cpu().detach().numpy()

            img_hat = np.moveaxis(img_hat, 0, 2)
            img = np.moveaxis(img, 0, 2)

            axes[0, n].imshow(img_hat)
            axes[0, n].set_xlabel('prediction of frame {}'.format(n))
            axes[1, n].imshow(img_hat)
            axes[1, n].set_xlabel('ground truth of frame {}'.format(n))

        break

    plt.tight_layout()
    plt.axis('off')
    plt.show()








if __name__ == '__main__':
    _main()
