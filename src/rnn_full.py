import csv
import json
import os
import numpy as np
from scipy.ndimage import uniform_filter1d
import torch
import torch.nn as nn
from torch.autograd import Variable

with open("rnn_params.json") as f:
    j = json.load(f)
    DATA_ORIG_DIR = j["data-orig-dir"]
    DATA_PARSED_DIR = j["data-parsed-dir"]
    DATASET_MULTIPLIER = j["dataset-multiplier"]
    TRAIN = j["train"]
    TRAIN_EPOCHS = j["train-epochs"]

"""
Function adds noise with additional uniform filter to the input array of values
:param y: input array of values
:param noise_range: optional range of added noise
:return: Array of resulting values with additional noise
"""
def add_noise(y, noise_range=(-0.4, 0.4)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=len(y))

    return y + uniform_filter1d(noise, size=3)


yParsed = []
for file in os.listdir(DATA_PARSED_DIR):
    with open(DATA_PARSED_DIR + "/" + file, "rb") as f:
        r_list = np.load(f).tolist()
        yParsed.append(r_list)

yOrig = []
for file in os.listdir(DATA_ORIG_DIR):
    with open(DATA_ORIG_DIR + "/" + file, "rb") as f:
        yOrig.append(np.load(f).tolist())

"""
Function creates a dataset sample using provided data. Sample index should be kept track of.
:param sample_ind: Current sample index
:return: Multiple values - input array, expected output array, original value array, sample index
"""
def gen_sample(sample_ind):
    if sample_ind > len(yParsed) - 1:
        sample_ind = 0
    s_out = yParsed[sample_ind]

    s_inp = add_noise(s_out)
    return s_inp, s_out, yOrig[sample_ind], sample_ind

"""
Function generates dataset with provided sample multiplier.
:param samples_per: How many samples will be generated per original file
:return: Multiple values - array of input arrays, array of expected arrays, array of original arrays
"""
def gen_dataset(samples_per):
    n = samples_per * len(yParsed)
    size = len(yParsed[0])

    set_inp = np.zeros((n, size))
    set_out = np.zeros((n, size))
    set_orig = np.zeros((n, size))

    sample_ind = 0
    for j in range(n):
        sample_inp, sample_out, sample_orig, sample_ind = gen_sample(sample_ind)
        set_inp[j, :] = sample_inp
        set_out[j, :] = sample_out
        set_orig[j, :] = sample_orig
        sample_ind += 1

    return set_inp, set_out, set_orig


data_inp, data_out, data_orig = gen_dataset(DATASET_MULTIPLIER)
split = int(len(data_inp) * 0.8)
train_inp, train_out = data_inp[:split], data_out[:split]
test_inp, test_out, test_orig = data_inp[split:], data_out[split:], data_orig[split:]

class Net(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super(Net, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden, batch_first=True)
        self.linear = nn.Linear(hidden, output_size)
        self.act = nn.Tanh()

    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        pred = self.act(self.linear(pred)).view(pred.data.shape[0], -1, 1)
        return pred


# If GPU is available, use it
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net = nn.DataParallel(Net(1, 33, 1)).to(device)
optimizer = torch.optim.Adam(net.parameters())
loss_f = nn.L1Loss()

if TRAIN:
    # Train network if required
    predictions = []

    for t in range(TRAIN_EPOCHS):
        hidden = None
        inp = Variable(torch.Tensor(train_inp.reshape((train_inp.shape[0], -1, 1))).to(device), requires_grad=True)
        out = Variable(torch.Tensor(train_out.reshape((train_out.shape[0], -1, 1))).to(device))
        pred = net(inp)
        optimizer.zero_grad()
        predictions.append(pred.cpu().data.numpy())
        loss = loss_f(pred, out)
        print(f"Loss after {t + 1} epoch: {loss}")
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), "model.pt")

else:
    # Load previously saved state
    net.load_state_dict(torch.load("model.pt"))

net.eval()
test_inp, test_out, test_orig = gen_dataset(1)
t_inp = Variable(torch.Tensor(test_orig.reshape((test_orig.shape[0], -1, 1))).to(device), requires_grad=True)
predicted_t = net(t_inp)

# Save results, in this case as csv file
to_save = range(len(predicted_t))
for i in to_save:
    w = open("results/" + str(i) + ".csv", "w", newline='')
    writer = csv.writer(w)

    predicted = predicted_t[i].cpu().data
    for j in range(len(predicted)):
        writer.writerow([predicted[j], test_out[i][j], test_orig[i][j]])
