#### ConfigUtils module

import os
import datetime
import ConfigParser

def init_configuration(config_file):
	# read config file
	config = ConfigParser.ConfigParser()
	if(not config.read(config_file)):
		raise IOError("Config file not found")
	# if Recovery section is present, use it as starting point
	# otherwise create a new backup folder and config file
	if('Recovery' in config.sections()):
		recovery_path = config.get('Recovery','Path')
		config_file_name = config_file
	else:
		recovery_path = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
		os.makedirs(recovery_path)
		print("Recovery folder: {}".format(recovery_path))
		config.add_section('Recovery')
		config.set('Recovery', 'Path', recovery_path)
		with open(os.path.join(recovery_path, 'config.ini'), 'wb') as recovery_config_file:
			config.write(recovery_config_file)
		config_file_name = recovery_config_file.name
	# create and return dict with net's options
	options = {
	    'hidden_layer_size': int(config.get('TrainingParameters','HiddenLayerSize')),
	    'weight_min_value': float(config.get('TrainingParameters','WeightsMin')),
	    'weight_max_value': float(config.get('TrainingParameters','WeightsMax')),
	    'max_epochs': int(config.get('TrainingParameters','MaxEpochs')),
	    'min_validation_logp_improvement': float(config.get('TrainingParameters','MinValidationLogpImprovement')),
	    'recovery_config_file': config_file_name,
	    'recovery_path': recovery_path,
	    'starting_epoch': int(config.get('Recovery','StartingEpoch')) if config.has_option('Recovery','StartingEpoch') else 0,
	    'learning_rate': float(config.get('Recovery','LearningRate')) if(config.has_option('Recovery','LearningRate')) else float(config.get('TrainingParameters','LearningRate')),
	    'logp_previous': float(config.get('Recovery','LogpPrevious')) if config.has_option('Recovery','LogpPrevious') else float("-inf"),
	    'learning_rate_divide': config.getboolean('Recovery', 'LearningRateDivide') if config.has_option('Recovery','LearningRateDivide') else False
	}
	return options

def log_current_epoch(epoch_num, learning_rate, logp_previous, learning_rate_divide, filename):
	config = ConfigParser.ConfigParser()
	config.read(filename)
	config.set('Recovery', 'StartingEpoch', epoch_num)
	config.set('Recovery', 'LearningRate', learning_rate)
	config.set('Recovery', 'LogpPrevious', logp_previous)
	config.set('Recovery', 'LearningRateDivide', learning_rate_divide)
	with open(filename, 'wb') as recovery_config_file:
		config.write(recovery_config_file)