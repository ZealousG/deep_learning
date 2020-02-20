"""
The initial vision is from https://morvanzhou.github.io/tutorials/, thank you!!!

Test environment:   Anaconda
                    PyCharm
                    CUDA+GPU

Last date:          2020-02-20

The programs need four extra Dependencies:
                    PyTorch Version:  1.3.1
                    Torchvision Version:  0.4.2
                    matplotlib
                    sklearn

"""

# -------------------------------------------------
# library
# -------------------------------------------------

# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import datasets, transforms
from matplotlib import cm
import matplotlib.pyplot as plt

# self library

# check library
try:
    # following function (plot_with_labels) is for visualization, can be ignored if not interested
    from sklearn.manifold import TSNE
    HAS_SK = True
    # check if CUDA have been install
    USE_CUDA = torch.cuda.is_available()
    # the CUDA can be used
    if USE_CUDA == True:
        device = torch.device("cuda")
except:
    if HAS_SK == False:
        print('Please install sklearn for layer visualization')
    if USE_CUDA == False:
        print('Please install CUDA for accelerating training')

print("PyTorch Version: ", torch.__version__)  # check PyTorch Version
print("Torchvision Version: ", torchvision.__version__)  # Torchvision Version:


# -------------------------------------------------
# Hyper Parameters
# -------------------------------------------------
EPOCH = 10  # Number of epochs to train for
BATCH_SIZE = 128  # Batch size for training (change depending on how much memory you have)
LR = 0.001  # Learning Rate is used for optimizers
DOWNLOAD_MNIST = False  # control download_MNIST
CLASS_NUM = 10

# Mnist digits dataset
if not (os.path.exists('./mnist_data/')) or not os.listdir('./mnist_data/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

# Hyper Parameters
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 256
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data

# Mnist digits dataset
if not (os.path.exists('../DATASET_public/mnist_data/')) or not os.listdir('../DATASET_public/mnist_data/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = datasets.MNIST(root='../DATASET_public/mnist_data/', train=True,
                            transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.MNIST(root='../DATASET_public/mnist_data/', train=False)
test_x = test_data.data.type(torch.FloatTensor).to(device)/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.targets.numpy()   # covert to numpy array

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])

        return out


model_rnn = RNN().to(device)
print(model_rnn)

optimizer = torch.optim.Adam(model_rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    print(epoch)
    for step, (data, target) in enumerate(train_loader):        # gives batch data
        b_x, b_y = data.to(device), target.to(device)
        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = model_rnn(b_x)  # rnn output

        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            test_output = model_rnn(test_x)  # (samples, time_step, input_size)
            test_output = test_output.cpu()
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            loss = loss.cpu()
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

# print 10 predictions from test data
# test_output = model_rnn(test_x[:10].view(-1, 28, 28))
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')


