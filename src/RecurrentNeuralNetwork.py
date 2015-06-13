#### RecurrentNeuralNetwork class
import os
import time

import numpy as np
import json
from collections import OrderedDict

import VocabularyUtils as vu
import ActivationFunctions as af
import ConfigUtils as cu

activation_hidden = af.sigmoid
activation_hidden_d = af.sigmoid_d
activation_output = af.softmax
activation_output_d = af.softmax_d

# return a n*m matrix of random float in [a,b) range
def get_random_matrix(n,m,a=-0.1,b=0.1):
    matrix = (b-a) * np.random.random([n,m]) + a
    matrix += (b-a) * np.random.random([n,m]) + a
    matrix += (b-a) * np.random.random([n,m]) + a
    return matrix

# Compute the dot of:
# - a (l+m)-length array, where the first l elements are in one-hot notation
# - a (l+m)*m matrix
# Output: m-length array
def sparse_dot(one_hot_index, l, not_sparse_array_section, m, matrix):
    not_sparse_section = np.dot(not_sparse_array_section, matrix[l:l+m,:])
    if(one_hot_index==-1):
        return not_sparse_section
    else:
        return np.add(matrix[one_hot_index,:], not_sparse_section)

class RecurrentNeuralNetwork:

    def __init__(self, vocabulary_filename, training_set_filename, validation_set_filename, test_set_filename, config_file):
        # load configuration file
        options = cu.init_configuration(config_file)
        self.recovery_config_file = options['recovery_config_file']
        self.recovery_path = options['recovery_path']
        # load dataset files
        self.training_set = open(training_set_filename, 'r')
        self.validation_set = open(validation_set_filename, 'r')
        self.test_set = open(test_set_filename, 'r')
        # load vocabulary
        v = open(vocabulary_filename, 'r')
        vocabulary = json.load(v)
        v.close()
        ss_occurrences = vocabulary.pop("</s>")
        vocabulary = OrderedDict({"</s>": ss_occurrences}.items() + OrderedDict(sorted(vocabulary.items(), key=lambda x: (-x[1], x[0]))).items())
        self.vocabulary_size = len(vocabulary)
        # net parameters
        self.learning_rate = options['learning_rate']
        self.starting_epoch = options['starting_epoch']
        self.max_epochs = options['max_epochs']
        self.min_improvement = options['min_validation_logp_improvement']
        self.logp_previous = options['logp_previous']
        self.learning_rate_divide = options['learning_rate_divide']
        # init neurons
        self.hidden_layer_size = options['hidden_layer_size']
        self.neu_input_index = -1
        self.neu_context = [0.1] * self.hidden_layer_size
        self.neu_hidden = np.zeros(self.hidden_layer_size)
        self.neu_output = np.zeros(self.vocabulary_size)
        # init synapses
        try:
            self.__restore_weights()
        except IOError:
            # weights between input and hidden layer
            self.syn_input = get_random_matrix(
                    self.vocabulary_size + self.hidden_layer_size, 
                    self.hidden_layer_size,
                    options['weight_min_value'],
                    options['weight_max_value']) 
            # weights between hidden and output layer
            self.syn_hidden = get_random_matrix(
                    self.hidden_layer_size, 
                    self.vocabulary_size,
                    options['weight_min_value'],
                    options['weight_max_value'])

    def __restore_weights(self):
        self.syn_input = np.load(os.path.join(self.recovery_path, 'input_hidden.npy'))
        self.syn_hidden = np.load(os.path.join(self.recovery_path, 'hidden_output.npy'))

    def __save_weights(self):
        np.save(os.path.join(self.recovery_path, "input_hidden"), self.syn_input)
        np.save(os.path.join(self.recovery_path, "hidden_output"), self.syn_hidden)

    def __init_previous_changes_for_momentum(self):
        self.hidden_deltas_previous = np.zeros(self.hidden_layer_size)
        self.change_input_not_sparse_previous = np.zeros([self.hidden_layer_size, self.hidden_layer_size])
        self.change_hidden_previous = np.zeros([self.hidden_layer_size, self.vocabulary_size])

    # reset context layer
    def __reset_context(self):
        self.neu_hidden = np.zeros(self.hidden_layer_size)
        self.neu_context = [0.1] * self.hidden_layer_size

    def __feedforward(self, previous_word):
        # hidden layer activations
        self.neu_hidden = activation_hidden(sparse_dot(previous_word, self.vocabulary_size, self.neu_context, self.hidden_layer_size, self.syn_input))
        # output layer activations
        self.neu_output = activation_output(np.dot(self.neu_hidden, self.syn_hidden))

    def __backpropagate(self, previous_word, current_word):
        if(current_word==-1):
            return
        # calculate error terms for output
        output_error = -1.0 * self.neu_output
        output_error[current_word] += 1.0
        output_deltas = output_error
        # calculate error terms for hidden
        hidden_error = np.dot(self.syn_hidden, output_deltas)
        hidden_deltas = np.multiply(activation_hidden_d(np.copy(self.neu_hidden)), hidden_error)
        # update hidden-->output weights
        change_hidden = np.outer(self.neu_hidden, output_deltas)
        self.syn_hidden += np.multiply(change_hidden, self.learning_rate) 
        # update input-->hidden weights
        if(previous_word!=-1):
            self.syn_input[previous_word,:] += np.multiply(hidden_deltas, self.learning_rate)
        change_input_not_sparse = np.outer(self.neu_context, hidden_deltas)
        self.syn_input[self.vocabulary_size:self.vocabulary_size+self.hidden_layer_size,:] += np.multiply(change_input_not_sparse, self.learning_rate)

    def train(self):
        for e in xrange(self.starting_epoch, self.max_epochs):
            # training phase
            self.__reset_context()
            #self.__init_previous_changes_for_momentum()
            self.training_set.seek(0)
            previous_word = 0
            for current_word in  self.training_set:
                current_word = int(current_word)
                self.__feedforward(previous_word)
                self.__backpropagate(previous_word, current_word) 
                self.neu_context = np.copy(self.neu_hidden)
                previous_word = current_word
            # validation phase
            self.__reset_context()
            self.validation_set.seek(0)
            previous_word = 0
            logp = 0
            word_counter = 0
            for current_word in  self.validation_set:
                current_word = int(current_word)
                self.__feedforward(previous_word)
                # word==-1 are not in vocabulary
                if(current_word!=-1):
                    word_counter += 1
                    logp += np.log10(self.neu_output[current_word])
                self.neu_context = np.copy(self.neu_hidden)
                previous_word = current_word
            # print progress
            print("*******************")
            print("{}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            print("Epoch {}".format(e))
            print("Learning rate {}".format(self.learning_rate))
            print("Validation log probability {}".format(logp))
            print("Validation words counter {}".format(word_counter))
            if(word_counter>0):
                print("Validation PPL {}".format(np.power(10.0, -logp / word_counter)))
            # check improvement
            if(logp < self.logp_previous):
                self.__restore_weights()
            else:
                self.__save_weights()
            if(logp*self.min_improvement < self.logp_previous):
                if (not self.learning_rate_divide): 
                    self.learning_rate_divide = True
                else:
                    break
            if (self.learning_rate_divide):
                self.learning_rate /= 2
            self.logp_previous = logp
            # log last epoch for recovery
            cu.log_current_epoch(e+1, self.learning_rate, self.logp_previous, self.learning_rate_divide, self.recovery_config_file)

    def __get_predicted_words(self):
        return self.neu_output.argmax()

    def test(self):
        self.__reset_context()
        self.test_set.seek(0)
        previous_word = 0
        logp = 0
        word_counter = 0
        error_counter = 0
        for current_word in  self.test_set:
            current_word = int(current_word)
            self.__feedforward(previous_word)
            if(current_word!=-1):
                word_counter += 1
                logp += np.log10(self.neu_output[current_word])
                if(current_word!=0 and current_word != self.__get_predicted_words()):
                    error_counter += 1
            self.neu_context = np.copy(self.neu_hidden)
            previous_word = current_word
        print("Test log probability {}".format(logp))
        print("Test words counter {}".format(word_counter))
        if(word_counter>0):
            print("Test PPL {}".format(np.power(10.0, -logp / word_counter)))
            print("Test WER {}".format(float(error_counter) / word_counter))