import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import random
import math
from itertools import cycle

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

def fom(pred, true):
    assert pred.shape == true.shape
    ae = np.absolute(pred-true)
    return (ae < 0.04).mean()

def fom_alpha(alpha, data):
    true_alpha = np.squeeze((data[:, :, 1] > 0).astype(np.float32))
    return np.mean(np.absolute(alpha-true_alpha))

def print_example(data, answer):
    idx1, idx2 = np.argsort(data[0, :, 1])[::-1][:2]
    print(data[0, max(0, idx1-2):idx1+2])
    print("...")
    print(data[0, idx2-2:min(idx2+2, 100)])
    print("...")
    print(answer[0])

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def vars_from_batch(input_batch, target_batch):
    target_batch = Variable(torch.FloatTensor(target_batch).contiguous().view(-1, 1))
    input_batch = Variable(torch.FloatTensor(input_batch))
    return input_batch, target_batch

def train_batch(input_variable, target_variable, net, 
          optimizer, criterion, use_cuda=False):

    optimizer.zero_grad()

    output, attention = net(input_variable)

    loss = criterion(output, target_variable)

    loss.backward()

    optimizer.step()

    return loss.data[0] / target_variable.shape[0]

def train(net, train_inputs, train_targets, epochs=1, batch_size=32, use_cuda=False, print_every=100):
    optimizer = optim.Adam(net.parameters())
    criterion = nn.MSELoss(size_average=False)
    
    idx = list(range((len(train_inputs))))
    
    batcher = cycle(idx)
    
    for epoch in range(1, epochs+1):
        print("Epoch %s of %s"%(epoch, epochs))
        loss = 0
        for batch in range(1, math.ceil(len(idx)/batch_size) + 1):
            batch_idx = [next(batcher) for i in range(batch_size)]
            batch_targets = train_targets[batch_idx]
            batch_inputs = train_inputs[batch_idx]
            batch_inputs, batch_targets = vars_from_batch(batch_inputs, batch_targets)
            loss += train_batch(batch_inputs, batch_targets, net, optimizer, criterion, use_cuda)
            if batch == 1 or (batch > 0 and batch % print_every == 0):
                print("Average batch loss over %s batchs: %.04f"%(batch, loss/batch))

def evaluate(batch_inputs, batch_targets, net, use_cuda=False):
    batch_inputs, batch_targets = vars_from_batch(batch_inputs, batch_targets)

    batch_outputs, batch_attentions = net(batch_inputs)

    return batch_outputs.data, batch_attentions.squeeze(-1).data

def fom(pred, true):
    assert pred.shape == true.shape
    ae = np.absolute(pred-true)
    return (ae < 0.04).mean()

def evaluate_randomly(inputs, targets, net, n=10):
    idx = np.random.randint(0, len(inputs), n)
    
    batch_inputs = inputs[idx]
    batch_targets = targets[idx]

    outputs, attentions = evaluate(batch_inputs, batch_targets, net, use_cuda=False)

    print("Accuracy: %.04f"%fom(np.array(outputs), batch_targets.reshape(-1, 1)))
    plt.style.use("seaborn-poster")
    plt.subplot(211)
    sns.heatmap(attentions, cbar=True, xticklabels=[])
    plt.title("Predicted Alpha")
    plt.subplot(212)
    sns.heatmap(np.squeeze(batch_inputs[:, :, 1] > 0).astype(np.float32)[:10], cbar=True)
    plt.title("True Alpha")
    plt.show()

def visualize(tweets, preds, alphas, targets, max_len):
    disp = ""
    start_tag = '''<span style="background-color:rgba(0,0,255,%0.2f);">'''
    end_tag = '''</span>'''
    for tweet, alpha_list, pred, target in zip(tweets, alphas, preds, targets):
        tokens = tweet.split()
        tokens = tokens[:max_len]
        for token, alpha in zip(tokens, alpha_list[:len(tokens)]):
            disp += start_tag%alpha + sanitize_token(token) + end_tag + " "
        disp += "%s %s %s <br>"%(tuple(zip(tuple(pred), tuple(target))))
    return disp   

def evaluate_randomly_disp_text(inputs, targets, net, n=10):
    idx = np.random.randint(0, len(inputs), n)
    
    batch_inputs = inputs[idx]
    batch_targets = targets[idx]

    outputs, attentions = evaluate(batch_inputs, batch_targets, net, use_cuda=False)
