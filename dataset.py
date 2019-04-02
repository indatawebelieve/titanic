import os
import pandas as pd
import numpy as np

def cuantify_sex(passengers):
	for i in range(0, len(passengers)):
		if passengers.at[i, 'Sex'] == 'male':
			passengers.at[i, 'Sex'] = 0
		elif passengers.at[i, 'Sex'] == 'female':
			passengers.at[i, 'Sex'] = 1

	return passengers

def clean_unnessesary_columns(passengers):
	return passengers.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

def clean_nan_values(passengers):
	for i in range(0, len(passengers)):
		if np.isnan(passengers.at[i, 'Age']):
			passengers = passengers.drop([i])
		elif np.isnan(passengers.at[i, 'Fare']):
			passengers = passengers.drop([i])

	return passengers

def split_logits_labels(passengers, df_size):
	labels = []
	for i in range(0, df_size):
		if i in passengers.index: # previous step has deleted some rows
			label = np.zeros(2)
			if passengers.at[i, 'Survived'] == 0:
				label[0] = 1.0
			elif passengers.at[i, 'Survived'] == 1:
				label[1] = 1.0
			labels.append(label)
	labels = np.array(labels)
	
	logits = passengers.drop(columns=['Survived'])
	logits = np.array(logits)
	
	return logits, labels

def get_trainds():
	passengers = pd.read_csv(os.path.join('.', 'input', 'train.csv'))
	entire_df_size = len(passengers)

	passengers = cuantify_sex(passengers)
	passengers = clean_unnessesary_columns(passengers)
	passengers = clean_nan_values(passengers)	

	return split_logits_labels(passengers, entire_df_size)

def get_testds():
	passengers = pd.read_csv(os.path.join('.', 'input', 'test.csv'))
	labels = pd.read_csv(os.path.join('.', 'input', 'gender_submission.csv'))
	entire_df_size = len(passengers)

	for i in range(0, entire_df_size):
		if passengers.at[i, 'PassengerId'] == labels.at[i, 'PassengerId']:
			passengers.at[i, 'Survived'] = labels.at[i, 'Survived']

	passengers = cuantify_sex(passengers)
	passengers = clean_unnessesary_columns(passengers)
	passengers = clean_nan_values(passengers)

	return split_logits_labels(passengers, entire_df_size)

