import numpy as np
from sklearn.model_selection import train_test_split
from load_dataset import load_dataset
from fullconnected import NeuralNetwork
import warnings
import matplotlib.pyplot as plt
#suppress warnings
warnings.filterwarnings('ignore')

a = load_dataset()
X, y, labels = a.get_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 15)

NN = NeuralNetwork(X_train.shape[1], hidden_size1= 8, hidden_size2= 6, output_size=7,lr=3e-10)
NN.fit(X_train, y_train, epochs = 50)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.plot(NN.losses)
plt.show()