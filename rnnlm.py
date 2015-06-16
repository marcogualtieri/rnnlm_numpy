import sys

import src.RecurrentNeuralNetwork as RNN
import src.VocabularyUtils as v

if __name__ == '__main__':

	if(len(sys.argv)>3):
		training = sys.argv[1]
		validation = sys.argv[2]
		test = None
		if(len(sys.argv)>3):
			test = sys.argv[3]
	else:
		print "Arguments: training validation test [config]"
		sys.exit(1)

	# config file
	config_file = sys.argv[4] if len(sys.argv)>4 else 'configs/default.ini'
	print("Using configuration file: {}".format(config_file))

	dataset = v.generate_dataset(training, validation, test)

	net = RNN.RecurrentNeuralNetwork(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], config_file)

	net.train()
	net.test()