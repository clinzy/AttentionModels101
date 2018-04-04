
"""
  Date Generation and Loading Code for the Date Normalization Task
  This code was adapted from ?
"""

import random
import json
import os
import csv
from faker import Faker
import babel
from babel.dates import format_date
import numpy as np

fake = Faker()
fake.seed(230517)
random.seed(230517)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY',
           ]

# change this if you want it to work with only a single language
LOCALES = ['en_US']
# LOCALES = babel.localedata.locale_identifiers()


def create_date():
    """
        Creates some fake dates 
        :returns: tuple containing 
                  1. human formatted string
                  2. machine formatted string
                  3. date object.
    """
    dt = fake.date_object()

    # wrapping this in a try catch because
    # the locale 'vo' and format 'full' will fail
    try:
        human = format_date(dt,
                            format=random.choice(FORMATS),
                            locale=random.choice(LOCALES))

        case_change = random.randint(0,3) # 1/2 chance of case change
        if case_change == 1:
            human = human.upper()
        elif case_change == 2:
            human = human.lower()

        machine = dt.isoformat()
    except AttributeError as e:
        # print(e)
        return None, None, None

    return human, machine, dt


def create_dataset(folder, dataset_name, n_examples, vocabulary=False):
    """
        Creates a csv dataset with n_examples and optional vocabulary
        :param dataset_name: name of the file to save as
        :n_examples: the number of examples to generate
        :vocabulary: if true, will also save the vocabulary
    """
    human_vocab = set()
    machine_vocab = set()

    with open(os.path.join(folder, dataset_name), 'w') as f:
        for i in range(n_examples):
            h, m, _ = create_date()
            if h is not None:
                f.write('"'+h + '","' + m + '"\n')
                human_vocab.update(tuple(h))
                machine_vocab.update(tuple(m))

    if vocabulary:
        human_vocab = ["<pad>",] + list(human_vocab)
        machine_vocab = ["<pad>",] + list(machine_vocab)

        int2human = dict(enumerate(human_vocab))
        int2human.update({len(int2human): '<eos>',
                          len(int2human)+1: '<sos>',
                          len(int2human)+2: '<unk>'})

        int2machine = dict(enumerate(machine_vocab))
        int2machine.update({len(int2machine):'<eos>',
                            len(int2machine)+1:'<sos>',
                            len(int2machine)+2: '<unk>'})

        human2int = {v: k for k, v in int2human.items()}
        machine2int = {v: k for k, v in int2machine.items()}

        with open(os.path.join(folder, 'human_vocab.json'), 'w') as f:
            json.dump(human2int, f)
        with open(os.path.join(folder, 'machine_vocab.json'), 'w') as f:
            json.dump(machine2int, f)

class Vocabulary(object):

    def __init__(self, vocabulary_file, padding=None):
        """
            Creates a vocabulary from a file
            :param vocabulary_file: the path to the vocabulary
        """
        self.vocabulary_file = vocabulary_file
        with open(vocabulary_file, 'r') as f:
            self.vocabulary = json.load(f)

        self.padding = padding
        self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

    def size(self):
        """
            Gets the size of the vocabulary
        """
        return len(self.vocabulary.keys())

    def string_to_int(self, text):
        """
            Converts a string into it's character integer 
            representation
            :param text: text to convert
        """
        characters = list(text)

        integers = []

        characters = ["<sos>",] + characters + ["<eos>",]

        for c in characters:
            if c in self.vocabulary:
                integers.append(self.vocabulary[c])
            else:
                integers.append(self.vocabulary['<unk>'])

        return integers

    def int_to_string(self, integers):
        """
            Decodes a list of integers
            into it's string representation
        """
        characters = []
        for i in integers:
            characters.append(self.reverse_vocabulary[i])

        return characters


class Data(object):

    def __init__(self, file_name, input_vocabulary, output_vocabulary):
        """
            Creates an object that gets data from a file
            :param file_name: name of the file to read from
            :param vocabulary: the Vocabulary object to use
            :param batch_size: the number of datapoints to return
            :param padding: the amount of padding to apply to 
                            a short string
        """

        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary
        self.file_name = file_name

    def load(self):
        """
            Loads data from a file
        """
        self.inputs = []
        self.targets = []

        with open(self.file_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.inputs.append(row[0])
                self.targets.append(row[1])

    def transform(self):
        """
            Transforms the data as necessary
        """
        # @TODO: use `pool.map_async` here?
        self.inputs = list(map(self.input_vocabulary.string_to_int, self.inputs))
        self.targets = list(map(self.output_vocabulary.string_to_int, self.targets))

    def generator(self, batch_size):
        """
            Creates a generator that can be used in `model.fit_generator()`
            Batches are generated randomly.
            :param batch_size: the number of instances to include per batch
        """
        instance_id = range(len(self.inputs))
        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)
                yield (np.array(self.inputs[batch_ids], dtype=int),
                       np.array(self.targets[batch_ids]))
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None

def create_and_load(folder, remake=False):
    train_fp = os.path.join(folder, "dates_train.csv")
    valid_fp = os.path.join(folder, "dates_valid.csv")
    input_vocab_fp = os.path.join(folder, 'human_vocab.json')
    output_vocab_fp = os.path.join(folder, 'machine_vocab.json')

    if remake:
        create_dataset(folder, "dates_train.csv", n_examples=48000, vocabulary=True)
        create_dataset(folder, "dates_valid.csv", n_examples=4800)

    input_vocab = Vocabulary(input_vocab_fp,)
    output_vocab = Vocabulary(output_vocab_fp,)

    training = Data(train_fp, input_vocab, output_vocab)
    validation = Data(valid_fp, input_vocab, output_vocab)

    training.load()
    validation.load()

    training.transform()
    validation.transform()
    train_pairs = list(zip(training.inputs, training.targets))
    val_pairs = list(zip(validation.inputs, validation.targets))

    return train_pairs, val_pairs, input_vocab, output_vocab