import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import random
import math
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

"""This code is largely adapted from the PyTorch official tutorial"""

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

def train(input_variable, target_variable, output_vocab, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, criterion, max_attn_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_attn_length, encoder.hidden_size))

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_hidden[0][0]

    decoder_input = Variable(torch.LongTensor([[output_vocab.size() - 1,]]))

    decoder_hidden = encoder_hidden

    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_variable[di])
        decoder_input = target_variable[di]  # Teacher forcing

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def variable_from_int_seq(sequence):
    return Variable(torch.LongTensor(sequence).view(-1, 1))

def train_iters(encoder, decoder, training_pairs, output_vocab, max_attn_length, 
                epochs=1, print_every=1000, plot_every=100, learning_rate=0.01):
    
    n_iters = len(training_pairs)
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs +1):
        print("Epoch %s of %s"%(epoch, epochs))
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_variable = variable_from_int_seq(training_pair[0])
            target_variable = variable_from_int_seq(training_pair[1])

            loss = train(input_variable, target_variable, output_vocab, encoder, decoder, 
                         encoder_optimizer, decoder_optimizer, criterion, max_attn_length)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    show_plot(plot_losses)
    
def evaluate(encoder, decoder, sequence, output_vocab, max_attn_length, use_cuda=False):
    input_variable = variable_from_int_seq(sequence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_attn_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_hidden[0][0]

    decoder_input = Variable(torch.LongTensor([[output_vocab.vocabulary["<sos>"],]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = np.zeros((max_attn_length, max_attn_length))

    for di in range(max_attn_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

        decoder_attentions[di] = np.array(decoder_attention.data)

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        if ni == output_vocab.vocabulary["<eos>"]:
            decoded_words.append('<eos>')
            break
        else:
            decoded_words.append(output_vocab.reverse_vocabulary[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions

def evaluate_randomly(encoder, decoder, pairs, input_vocab, output_vocab, max_attn_len, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        input_words = input_vocab.int_to_string(pair[0])
        print('>', "".join(input_words))
        print('=', "".join(output_vocab.int_to_string(pair[1])))
        output_words, attentions = evaluate(encoder, decoder, pair[0], output_vocab, max_attn_len)
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print('')

        end_of_output = output_words.index("<eos>") + 1 if "<eos>" in output_words else len(output_words)

        end_of_input = input_words.index("<eos>") + 1 if "<eos>" in input_words else len(input_words)

        output_words = output_words[:end_of_output]
        input_words = input_words[:end_of_input]

        attentions = attentions[:end_of_output, :end_of_input]

        plt.style.use("seaborn-poster")

        sns.heatmap(attentions,  cmap='bone', xticklabels=input_words, yticklabels=output_words)

        plt.show()