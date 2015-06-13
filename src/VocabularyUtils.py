#### VocabularyUtils module

import re
import json
import sys
from collections import OrderedDict
import os.path as osp

def __words_from_file(filename):
	# clean file from <doc></doc> tags and load text
	raw_text = ""
	with open(filename) as f:
		for line in f:
			if(not '<doc id' in line and not '</doc>' in line):
				raw_text += line + " </s> "
	# clean punctuation (leave apostrophe in case like it's, don't...)
	text = " ".join(re.findall("[a-zA-Z]+'[a-zA-Z]+|[a-zA-Z]+|<\/s>", raw_text))
	#text = re.sub("(?<![a-zA-Z])'|'(?![a-zA-Z])", " ", text)
	# lower case 
	text = text.lower()
	# tokenization (including trim/strip)
	words = text.split()
	return words

def __save_set_in_1hot(vocabulary_keys, words, filename):
	with open(filename, 'w') as f:
		for i in range(0, len(words)):
			word = words[i]
			index = -1
			if word in vocabulary_keys:
				index = vocabulary_keys.index(word)
			print>>f, index

def generate_dataset(training_set_filename, validation_set_filename, test_set_filename):
	dataset_path = osp.dirname(training_set_filename)
	training_words = __words_from_file(training_set_filename)
	# create vocabulary from training set 
	# as a dictionary where key: word, value: occurrences
	vocabulary = dict()
	for word in training_words:
		if word in vocabulary:
			vocabulary[word] += 1
		else:
			vocabulary[word] = 1
	ss_occurrences = vocabulary.pop("</s>")
	vocabulary = OrderedDict({"</s>": ss_occurrences}.items() + OrderedDict(sorted(vocabulary.items(), key=lambda x: (-x[1], x[0]))).items())
	vocabulary_keys = vocabulary.keys()
	# generate vocabulary file
	vocabulary_filename = osp.join(dataset_path, 'vocabulary')
	json.dump(vocabulary, open(vocabulary_filename,'w'), indent=4)
	# generate training set file
	training_set_1hot_filename = osp.join(dataset_path, 'training1hot')
	__save_set_in_1hot(vocabulary_keys, training_words, training_set_1hot_filename)
	# generate validation set file
	validation_set_1hot_filename = osp.join(dataset_path, 'validation1hot')
	__save_set_in_1hot(vocabulary_keys, __words_from_file(validation_set_filename), validation_set_1hot_filename)
	# generate test set file
	test_set_1hot_filename = osp.join(dataset_path, 'test1hot')
	__save_set_in_1hot(vocabulary_keys, __words_from_file(test_set_filename), test_set_1hot_filename)

	return [vocabulary_filename, training_set_1hot_filename, validation_set_1hot_filename, test_set_1hot_filename]