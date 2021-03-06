import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

from convLSTM_slip_detection_1layer import ConvLSTMCell as convLSTMDetect
from convLSTM_frame_pred import ConvLSTMCell as convLSTMPred, random_split_customized

# import dataset loader module
from convLSTM_dataset import *

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


import IPython


class ConvLSTMChained(nn.Module):

    def __init__(self, n_frames_ahead=2, n_frames=11):
        super(ConvLSTMChained, self).__init__()
        self.n_frames_ahead = n_frames_ahead
        self.n_frames = n_frames
        self.channels = 2
        self.pred_net = convLSTMPred(self.channels, 32, self.n_frames_ahead)
        self.detect_net = convLSTMDetect(self.channels, 64, n_class=4)

        self.output_list = {'pred': [], 'detect': []}

    def forward(self, t, input, prev): # prev defined as a dict
        prev_p = prev['pred']
        prev_d = prev['detect']
        if t < self.n_frames_ahead - 1:
            out_p, prev_p = self.pred_net(input, prev_p)
            prev = {'pred': prev_p, 'detect': prev_d}
            out_d = None
        elif t < self.n_frames - 1:
            # print '[INFO] forwarding: time frame {}'.format(t)
            out_p, prev_p = self.pred_net(input, prev_p)
            out_d, prev_d = self.detect_net(input, prev_d)

            prev = {'pred': prev_p, 'detect': prev_d}

            self.output_list['pred'].append(out_p)
            self.output_list['detect'].append(out_d)

            # print 'prev state size {}'.format(len(prev['pred']))

        else:

            out_d, prev_d = self.detect_net(self.output_list['pred'][self.n_frames_ahead - (t-(self.n_frames-1))], prev_d)
            prev['detect'] = prev_d
            self.output_list['detect'].append(out_d)

        return out_d, prev


def load_state_dict(model, path_list):

    model_dict = model.state_dict()
    # for key, value in model_dict.iteritems():
    #     print key

    for type_key, path in path_list.iteritems():
        # print '-----------------------------'
        pretrained_dict = torch.load(path)
        # for key, value in pretrained_dict.iteritems():
        #     print key

        # 1. filter out unnecessary keys
        pretrained_dict = {(type_key + '.' + k): v for k, v in pretrained_dict.items() if (type_key + '.' + k) in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        # print '-----------------------------'



def show_model_size(model, input_size):
    # Estimate Size
    # from pytorch_modelsize.pytorch_modelsize import SizeEstimator
    #
    # se = SizeEstimator(model, input_size=input_size)
    # print(se.estimate_size())

    # Returns
    # (size in megabytes, size in bits)
    # (408.2833251953125, 3424928768)

    print(se.param_bits)  # bits taken up by parameters
    print(se.forward_backward_bits)  # bits stored for forward and backward
    print(se.input_bits)  # bits for input

def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    batch_size, channels, height, width = 64, 2, 30, 30
    hidden_size = 64 # 64           # hidden state size
    lr = 1e-5     # learning rate
    n_frames = 11           # sequence length
    max_epoch = 30  # number of epochs

    convlstm_dataset = convLSTM_Dataset_dxdy(dataset_dir='../dataset/resample_skipping_stride1',
                                        n_class=4,
                                        transform=transforms.Compose([
                                            RandomHorizontalFlip(),
                                            RandomVerticalFlip(),
                                            ToTensor(),
                                        ])
                                        )


    train_sampler, test_sampler = random_split_customized(convlstm_dataset, train_ratio=0.9)

    train_dataloader = DataLoader(convlstm_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=4)
    test_dataloader = DataLoader(convlstm_dataset, batch_size=batch_size, sampler=test_sampler,
                                 num_workers=4)

    test_size = len(test_sampler)
    for n_frames_ahead in range(1, 6):
        print('Instantiating model.............')
        model = ConvLSTMChained(n_frames_ahead=n_frames_ahead, n_frames=n_frames)
        print(repr(model))

        # print model.state_dict()

        # load pretrained_model_diction
        path_pred = './saved_model/convlstm_frame_predict_20190415_400epochs_4000data_flipped_{}f_ahead.pth'.format(n_frames_ahead)
        path_detect = './saved_model/convlstm__model_1layer_augmented_11frames_400epochs_20190415.pth'

        path_dict = {'pred_net': path_pred, 'detect_net': path_detect}

        load_state_dict(model, path_dict)

        # IPython.embed()

        if torch.cuda.is_available():
            # print 'sending model to GPU'
            model = model.cuda()

        print('Create input and target Variables')
        x = Variable(torch.rand(n_frames, batch_size, channels, height, width))
        # y = Variable(torch.randn(T, b, d, h, w))
        y = Variable(torch.rand(batch_size))

        print('Create a MSE criterion')
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

        # IPython.embed()




        import time

        model = model.eval()

        test_loss = 0
        n_right = 0

        start = time.time()
        for test_step, test_sample_batched in enumerate(test_dataloader):

            start = time.time()

            model.output_list = {'pred': [], 'detect': []}

            x = test_sample_batched['frames']
            y = test_sample_batched['target']
            x = torch.transpose(x, 0, 1)
            # x = x.type(torch.FloatTensor)

            if torch.cuda.is_available():
                # print 'sending input and target to GPU'
                x = x.type(torch.cuda.FloatTensor)
                y = y.type(torch.cuda.FloatTensor)

            prev = {'pred': None, 'detect': None}

            for t in range(0, n_frames):
                out_test,  prev = model(t, x[t], prev)

            y = y.long()

            test_loss += loss_fn(out_test, y).item() * batch_size
            # Compute accuracy
            _, argmax_test = torch.max(out_test, 1)
            # print argmax_test
            # print y
            n_right += sum(y == argmax_test.squeeze()).item()

            # print '[TIME] the forward time: {}'.format(time.time() - start)
            # print n_right
        test_loss_reduced = test_loss / test_size
        test_accuracy = float(n_right) / test_size

        print ('[ TEST set] Step {}, Loss: {:.6f}, Acc: {:.4f}'.format(
            test_step + 1, test_loss_reduced, test_accuracy))




if __name__ == '__main__':
    _main()
