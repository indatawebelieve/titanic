import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import get_trainds, get_testds

logits, labels = get_trainds()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(500, activation='relu', input_shape=[6,]))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(logits, labels, epochs=40, batch_size=5, validation_split=0.2)

'''
hist = pd.DataFrame(history.history)

print(hist['acc'].describe())
print(hist['loss'].describe())

plt.figure()
acc, = plt.plot(hist.index.tolist(), hist['acc'], label='Accuracy')
loss, = plt.plot(hist.index.tolist(), hist['loss'], label='Loss')
plt.xlabel('Epochs')
plt.legend(handles=[acc, loss])
plt.show()
'''

test_x, test_y = get_testds()
test_loss, test_acc = model.evaluate(test_x, test_y)

print('Test accuracy:', test_acc)

