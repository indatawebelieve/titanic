import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(os.path.join('.', 'input', 'train.csv'))

# 90% payed 77 or less, 95% payed 112 or less. This is confirmed by median
print(df['Fare'].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95]))
print(df['Age'].describe())

plt.hist(df['Age'])
plt.show()

# Here we can see a big difference between male and female, more males
plt.pie(df.groupby(['Sex']).size(), autopct='%1.1f%%', labels=['Female', 'Male'])
plt.show()

# The most of the people payed cheaper fares
plt.hist(df['Fare'])
plt.xlabel('Payed fare')
plt.ylabel('Frequency')
plt.show()

''' Some people of Upper class payed more, but also there are others 
that payed the same as middle and lower class which is extrange
supposition: people that payed more and belogs to upper class have family
therefore fare could be a derived variable from Pclass and SibSp and Parch '''
plt.scatter(df['Fare'], df['Pclass'])
plt.xlabel('Payed Fare')
plt.ylabel('Class')
plt.text(450, 3, '1- Upper')
plt.text(450, 2.9, '2- Middle')
plt.text(450, 2.8, '3- Lower')
plt.show()

for i in range(0, len(df)):
	suma = int(df.at[i, 'SibSp']) + int(df.at[i, 'Parch'])
	df.at[i, 'SibSpParch'] = suma

# Waiting for a linear regression to confirm previous supposition, but didnt happen
plt.scatter(df['SibSpParch'], df['Fare'])
plt.show()

# Most of people on the lower class, middle and upper are balanced
plt.pie(df.groupby(['Pclass']).size(), autopct='%1.1f%%', labels=['1-Upper', '2-Middle', '3-Lower'])
plt.show()

