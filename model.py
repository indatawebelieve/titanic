import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import get_trainds, get_testds

num_epochs = 60
batch_size = 5
num_trainings = 1
logits, labels = get_trainds()
test_x, test_y = get_testds()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(500, activation='relu', input_shape=[7,]))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

train_acc_sum = 0.0
val_acc_sum = 0.0
test_acc_sum = 0.0
for _ in range(0, num_trainings):
	history = model.fit(logits, labels, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)

	hist = pd.DataFrame(history.history)

	train_acc_sum += hist['acc'].mean()
	val_acc_sum += hist['val_acc'].mean()

	test_loss, test_acc = model.evaluate(test_x, test_y)

	test_acc_sum += test_acc

	plt.figure()
	acc, = plt.plot(hist.index.tolist(), hist['acc'], label='Accuracy')
	loss, = plt.plot(hist.index.tolist(), hist['loss'], label='Loss')
	plt.xlabel('Epochs')
	plt.legend(handles=[acc, loss])
	plt.show()

print('------------------------------------------------')
print('Accuracies:')
print('Training mean accuracy: ' + str(train_acc_sum/num_trainings))
print('Validation mean accuracy: ' + str(val_acc_sum/num_trainings))
print('Testing mean accuracy: ' + str(test_acc_sum/num_trainings))

submission = pd.read_csv(os.path.join('.', 'input', 'gender_submission.csv'))
predictions = model.predict(test_x)
processedpredictions = []
for prediction in predictions:
	survived = np.argmax(prediction)
	processedpredictions.append(survived)

submission['Survived'] = processedpredictions
submission.to_csv(os.path.join('.', 'output', 'my_2nd_submission.csv'), index=False)
