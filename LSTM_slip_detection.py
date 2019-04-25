import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


class LSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, n_class, n_frames=11):
        super(LSTMCell, self).__init__()
        self.input_size = input_size # here it refers to embedding_dim the size of
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

        self.dp1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(self.hidden_size, 256)

        self.dp2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, n_class)

        self.n_frames = n_frames

    def forward(self, input_, hc, out_bool):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]

        # generate empty prev_state, if None is provided
        if hc is None:
            state_size = [batch_size, self.hidden_size]  # (B,Hidden_size)
            hc = (  # hidden and cell
                Variable(torch.zeros(state_size)).type(torch.cuda.FloatTensor),
                Variable(torch.zeros(state_size)).type(torch.cuda.FloatTensor)
            )  # list of h[t-1] and C[t-1]: both of size [batch_size, hidden_size, D, D]

        # data size is [batch, channel, height, width]

        hc = self.lstm(input_, hc)

        h, c = hc
        # print h.size
        # IPython.embed()
        out = None
        if out_bool:
            flat = h.view(-1, h.size()[-1])
            # print flat.size()
            out = self.linear1(self.dp1(flat))
            out = self.linear2(self.dp2(out))

        return out, hc


import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

import IPython

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # IPython.embed()
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax




from convLSTM_dataset import *
from torch.utils.data import DataLoader

from torch.utils.data.dataset import random_split

# from logger import Logger
# logger = Logger('./logs')
def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    batch_size, feature_size = 32, 1800
    n_class = 4
    hidden_size = 1024 # 64           # hidden state size
    lr = 1e-5     # learning rate
    n_frames = 11           # sequence length
    N_dataset_frames = 15
    max_epoch = 50  # number of epochs

    lstm_dataset = LSTM_Dataset(dataset_dir='../dataset/resample_skipping_stride1',
                                n_class=4,
                                transform=transforms.Compose([
                                                                ToTensor()])
                                )
    train_ratio = 0.9
    train_size = int(train_ratio*len(lstm_dataset))
    test_size = len(lstm_dataset) - train_size

    print('=== dataset size, train size, test size:===')
    print len(lstm_dataset), train_size, test_size

    train_dataset, test_dataset = random_split(lstm_dataset, [train_size, test_size])


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=1)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # set manual seed
    # torch.manual_seed(0)

    print('Instantiate model................')
    model = LSTMCell(feature_size, hidden_size, n_class=n_class, n_frames=11)
    print(repr(model))

    if torch.cuda.is_available():
        # print 'sending model to GPU'
        model = model.cuda()

    print('Create input and target Variables')
    x = Variable(torch.rand(n_frames, batch_size, feature_size))
    # y = Variable(torch.randn(T, b, d, h, w))
    y = Variable(torch.rand(batch_size))

    print('Create a MSE criterion')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    # IPython.embed()

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        loss_train = 0
        n_right_train = 0
        for step, sample_batched in enumerate(train_dataloader):

            model = model.train()

            x = sample_batched['frames']
            y = sample_batched['target']
            x = torch.transpose(x, 0, 1)  # transpose time sequence and batch (N, batch, feature_size)
            # x = x.type(torch.FloatTensor)
            # print x.size()

            if torch.cuda.is_available():
                # print 'sending input and target to GPU'
                x = x.type(torch.cuda.FloatTensor)
                y = y.type(torch.cuda.FloatTensor)

            state = None

            for t in range(N_dataset_frames-n_frames, N_dataset_frames):
                # print x[t,0,0,:,:]
                out_bool = True if t == N_dataset_frames - 1 else False
                out, state = model(x[t], state, out_bool)

            # out = out.long()
            y = y.long()


            # print out.size(), y.size()
            loss = loss_fn(out, y)
            # print(' > Epoch {:2d} loss: {:.7f}'.format((epoch+1), loss.data[0]))

            # zero grad parameters
            model.zero_grad()

            # compute new grad parameters through time!
            loss.backward()
            optimizer.step()
            # learning_rate step against the gradient
            # optimizer.step()
            #  for p in model.parameters():
            #     p.data.sub_(p.grad.data * lr)

            loss_train += loss.item()*batch_size
            # Compute accuracy

            _, argmax = torch.max(out, 1)
            # print y, argmax.squeeze()
            # accuracy = (y == argmax.squeeze()).float().mean() # accuracy in each batch
            n_right_train += sum(y == argmax.squeeze()).item()

            N_step_vis = 20
            if (step + 1) % N_step_vis == 0:
                loss_train_reduced = loss_train / (N_step_vis*batch_size)
                train_accuracy = float(n_right_train) / (N_step_vis*batch_size)
                loss_train = 0
                n_right_train = 0
                print '=================================================================='
                print ('[TRAIN set] Epoch {}, Step {}, Loss: {:.6f}, Acc: {:.4f}'
                       .format(epoch, step + 1, loss_train_reduced, train_accuracy))



                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                model = model.eval()

                test_loss = 0
                n_right = 0
                for test_step, test_sample_batched in enumerate(test_dataloader):
                    x = test_sample_batched['frames']
                    y = test_sample_batched['target']
                    x = torch.transpose(x, 0, 1)
                    # x = x.type(torch.FloatTensor)

                    if torch.cuda.is_available():
                        # print 'sending input and target to GPU'
                        x = x.type(torch.cuda.FloatTensor)
                        y = y.type(torch.cuda.FloatTensor)

                    state_test = None

                    for t in range(N_dataset_frames - n_frames, N_dataset_frames):
                        # print x[t,0,0,:,:]
                        out_bool = True if t == N_dataset_frames - 1 else False
                        out_test, state_test = model(x[t], state_test, out_bool)

                    # out = out.long()
                    y = y.long()

                    # print out.size(), y.size()
                    test_loss += loss_fn(out_test, y).item() * batch_size

                    # Compute accuracy
                    _, argmax_test = torch.max(out_test, 1)
                    # print argmax_test
                    # print y
                    n_right += sum(y == argmax_test.squeeze()).item()

                # print n_right
                test_loss_reduced = test_loss/test_size
                test_accuracy = float(n_right)/test_size




                # print test_accuracy
                print ('[ TEST set] Epoch {}, Step {}, Loss: {:.6f}, Acc: {:.4f}'
                       .format(epoch, step + 1, test_loss_reduced, test_accuracy))
                # 1. Log scalar values (scalar summary)
                # info = {'loss': loss_train_reduced, 'accuracy': train_accuracy,
                #         'test_loss': test_loss_reduced, 'test_accuracy': test_accuracy}
                #
                # for tag, value in info.items():
                #     logger.scalar_summary(tag, value, epoch*(train_size/batch_size) + (step + 1))
                #
                # # 2. Log values and gradients of the parameters (histogram summary)
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('.', '/')
                #     logger.histo_summary(tag, value.data.cpu().numpy(), epoch*(train_size/batch_size) + (step + 1))
                #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch*(train_size/batch_size) + (step + 1))

                # 3. Log training images (image summary)
                # info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

                # for tag, images in info.items():
                #     logger.image_summary(tag, images, step + 1)





    start = time.time()
    y_true = []
    y_pred = []
    for test_step, test_sample_batched in enumerate(test_dataloader):
        x = test_sample_batched['frames']
        y = test_sample_batched['target']
        x = torch.transpose(x, 0, 1)
        # x = x.type(torch.FloatTensor)

        if torch.cuda.is_available():
            # print 'sending input and target to GPU'
            x = x.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)

        state_test = None
        out_test = None

        for t in range(N_dataset_frames - n_frames, N_dataset_frames):
            # print x[t,0,0,:,:]
            out_bool = True if t == N_dataset_frames - 1 else False
            out_test, state_test = model(x[t], state_test, out_bool)

        _, argmax_test = torch.max(out_test, 1)

        y_true.append(y.cpu().numpy().astype(int))
        y_pred.append(argmax_test.cpu().numpy())
    #     print 'show a batch in test set:'
    #     print y
    #     print argmax_test.squeeze()
    #     break
    # print 'one batch inference time:', (time.time() - start)/batch_size
    # save the trained model parameters
    # print y_true
    # IPython.embed()
    y_true = np.concatenate((np.ravel(np.array(y_true[:-1])), np.array(y_true[-1])), axis=0)
    y_pred = np.concatenate((np.ravel(np.array(y_pred[:-1])), np.array(y_pred[-1])), axis=0)
    # IPython.embed()

    print np.array(y_true)
    print np.array(y_pred)

    torch.save(model.state_dict(), './saved_model/lstm_model_1layer_11frames_50epochs_20190425.pth') # arbitrary file extension

    ## calculate metric scores
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print '------------------------------------'
    print 'accuracy: {}'.format(accuracy)
    print 'precision: {}'.format(precision)
    print 'recall: {}'.format(recall)
    print 'f1_score: {}'.format(f1_score)
    print 'support: {}'.format(support)

    # Plot normalized confusion matrix
    class_names = ['Translational slip', 'Rotional slip', 'Rolling', 'Stable']
    plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,
                          title='Normalized Confusion Matrix')
    plt.show()


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


if __name__ == '__main__':
    _main()
