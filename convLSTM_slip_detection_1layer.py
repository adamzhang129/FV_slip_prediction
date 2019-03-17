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

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

        # ker2_size = int(0.25*hidden_size)
        # self.Conv = nn.Conv2d(hidden_size, ker2_size, KERNEL_SIZE, padding=PADDING)

        self.height, self.width = 30, 30
        self.linear1 = nn.Linear(self.hidden_size*self.height*self.width, 256)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 2)


        # self.softmax = nn.Softmax()


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # print spatial_size

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)  # (B,Hidden_size, H, W)
            prev_state = (
                Variable(torch.zeros(state_size)).type(torch.cuda.FloatTensor),
                Variable(torch.zeros(state_size)).type(torch.cuda.FloatTensor)
            )  # list of h[t-1] and C[t-1]: both of size [batch_size, hidden_size, D, D]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        # print input_.type(), prev_hidden.type()
        stacked_inputs = torch.cat((input_, prev_hidden), 1)  # concat x[t] with h[t-1]
        gates = self.Gates(stacked_inputs)

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
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        # print hidden.size()
        # conv2 = self.Conv(hidden)
        # flat = conv2.view(-1, conv2.size(1) * conv2.size(2) * conv2.size(3))

        flat = hidden.view(-1, hidden.size(1)*hidden.size(2)*hidden.size(3))
        # print flat.size()
        out = self.linear1(flat)
        out = self.dropout(out)
        out = self.linear2(out)

        return out,  (hidden, cell)


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
    batch_size, channels, height, width = 32, 3, 30, 30
    hidden_size = 64 # 64           # hidden state size
    lr = 1e-5     # learning rate
    n_frames = 10           # sequence length
    max_epoch = 1  # number of epochs

    convlstm_dataset = convLSTM_Dataset(dataset_dir='../dataset/resample_skipping',
                                        n_class=2,
                                        transform=transforms.Compose([
                                            RandomHorizontalFlip(),
                                            RandomVerticalFlip(),
                                            ToTensor(),
                                        ])
                                        )
    train_ratio = 0.9
    train_size = int(train_ratio*len(convlstm_dataset))
    test_size = len(convlstm_dataset) - train_size

    train_dataset, test_dataset = random_split(convlstm_dataset, [train_size, test_size])


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4)

    # set manual seed
    # torch.manual_seed(0)

    print('Instantiate model')
    model = ConvLSTMCell(channels, hidden_size)
    print(repr(model))

    if torch.cuda.is_available():
        # print 'sending model to GPU'
        model = model.cuda()

    print('Create input and target Variables')
    x = Variable(torch.rand(n_frames, batch_size, channels, height, width))
    # y = Variable(torch.randn(T, b, d, h, w))
    y = Variable(torch.rand(batch_size))

    print('Create a MSE criterion')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)



    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        loss_train = 0
        n_right_train = 0
        for step, sample_batched in enumerate(train_dataloader):

            model = model.train()

            x = sample_batched['frames']
            y = sample_batched['target']
            x = torch.transpose(x, 0, 1)  # transpose time sequence and batch (N, batch, channel, height, width)
            # x = x.type(torch.FloatTensor)
            # print x.size()

            if torch.cuda.is_available():
                # print 'sending input and target to GPU'
                x = x.type(torch.cuda.FloatTensor)
                y = y.type(torch.cuda.FloatTensor)

            state = None
            out = None


            for t in range(0, n_frames):
                # print x[t,0,0,:,:]
                out, state = model(x[t], state)
                # loss += loss_fn(state[0], y[t])

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

            if (step + 1) % 50 == 0:
                loss_train_reduced = loss_train / (50*batch_size)
                train_accuracy = float(n_right_train) / (50*batch_size)
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
                    out_test = None

                    for t in range(0, n_frames):
                        out_test, state_test = model(x[t], state_test)
                        # loss += loss_fn(state[0], y[t])

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

        for t in range(0, n_frames):
            out_test,  state_test = model(x[t], state_test)

        _, argmax_test = torch.max(out_test, 1)

        y_true.append(y.cpu().numpy())
        y_pred.append(argmax_test.cpu().numpy())
    #     print 'show a batch in test set:'
    #     print y
    #     print argmax_test.squeeze()
    #     break
    # print 'one batch inference time:', (time.time() - start)/batch_size
    # save the trained model parameters
    y_true = np.array(y_true).astype(int)
    y_true = np.ravel(y_true)
    y_pred = np.ravel(np.array(y_pred))

    print np.array(y_true)
    print np.array(y_pred)

    torch.save(model.state_dict(), './saved_model/convlstm__model_1layer_augmented_20190315.pth') # arbitrary file extension

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
    class_names = ['slip', 'nonslip']
    plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()

    


if __name__ == '__main__':
    _main()
