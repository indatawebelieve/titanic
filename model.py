import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
df = pd.read_csv(os.path.join('.', 'input', 'test.csv'))
dflabels = pd.read_csv(os.path.join('.', 'input', 'gender_submission.csv'))

for i in range(0, len(df)):
	if int(df.at[i, 'PassengerId']) == int(dflabels.at[i, 'PassengerId']):
		df.at[i, 'Survived'] = dflabels.at[i, 'Survived'] 
'''

passengers = pd.read_csv(os.path.join('.', 'input', 'train.csv'))
entire_df_size = len(passengers)

def cuantify_sex(passengers):
	for i in range(0, len(passengers)):
		if passengers.at[i, 'Sex'] == 'male':
			passengers.at[i, 'Sex'] = 0
		elif passengers.at[i, 'Sex'] == 'female':
			passengers.at[i, 'Sex'] = 1

	return passengers

def clean_unnessesary_columns(passengers):
	return passengers.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

def clean_nan_ages(passengers):
	for i in range(0, len(passengers)):
		if np.isnan(passengers.at[i, 'Age']):
			passengers = passengers.drop([i])

	return passengers

def split_logits_labels(passengers):
	labels = []
	for i in range(0, entire_df_size):
		if i in passengers.index:
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

passengers = cuantify_sex(passengers)
passengers = clean_unnessesary_columns(passengers)
passengers = clean_nan_ages(passengers)		
logits, labels = split_logits_labels(passengers)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(48, activation='relu', input_shape=[6,]))
model.add(tf.keras.layers.Dense(36, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
#print(model.summary())

history = model.fit(logits, labels, epochs=20)

hist = pd.DataFrame(history.history)
#print(hist.head())

plt.figure()
acc, = plt.plot(hist.index.tolist(), hist['acc'], label='Accuracy')
loss, = plt.plot(hist.index.tolist(), hist['loss'], label='Loss')
plt.xlabel('Epochs')
plt.legend(handles=[acc, loss])
plt.show()



