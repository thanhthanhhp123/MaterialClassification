import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

class load_dataset():
	def __init__(self):
		pass
	def get_data(self):
		file_path = 'files'
		files = os.listdir(file_path)

		step = 5
		window = 11

		X = []; y = []
		for file in files:
			data = pd.read_csv(file_path + '/' + file, header = None).to_numpy()
			data = data[:56]

			N = (data.shape[0] - window) // step + 1      

			for i in range(N):
				X.append(data[i * step : i * step + window])
				y.append(file.split('_')[0])

		X = np.array(X, dtype = float)
		y = np.array(y)	

		return X, y

	def normalize(self, X):
		mu = np.mean(X, axis = 0)
		sigma = np.std(X, axis = 0)

		return (X - mu) / sigma

	def get_dataset(self):
		X, y = self.get_data()

		X = X.reshape(X.shape[0], -1)
		X = self.normalize(X)

		le = LabelEncoder()
		le.fit(y)

		y = le.transform(y)
		labels = le.inverse_transform(np.arange(0, y.max() + 1))

		y = np.array(y, ndmin = 2).T

		# print('Shape of X: ', X.shape)
		# print('Shape of y: ', y.shape)
		# print('Labels: ', labels)

		return X, y, labels	
