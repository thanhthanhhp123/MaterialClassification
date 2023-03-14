import numpy as np
class NeuralNetwork:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        self.w1 = np.random.randn(n_input, n_hidden1) / np.sqrt(n_input)
        self.b1 = np.zeros((1, n_hidden1))
        self.w2 = np.random.randn(n_hidden1, n_hidden2) / np.sqrt(n_hidden1)
        self.b2 = np.zeros((1, n_hidden2))
        self.w3 = np.random.randn(n_hidden2, n_output) / np.sqrt(n_hidden2)
        self.b3 = np.zeros((1, n_output))

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def calculate_loss(self, X, y):
        z1 = np.dot(X, self.w1) + self.b1
        self.activation1 = self.relu(z1)
        z2 = np.dot(self.activation1, self.w2) + self.b2
        self.activation2 = self.relu(z2)
        z3 = np.dot(self.activation2, self.w3) + self.b3
        y_hat = self.softmax(z3)
        loss = -np.mean(y * np.log(y_hat))
        return loss

    def backward(self, X, y, y_hat, learning_rate):
        delta3 = y_hat - y
        dw3 = np.dot(self.activation2.T, delta3)
        db3 = np.sum(delta3, axis=0)

        delta2 = np.dot(delta3, self.w3.T) * (self.activation2 > 0)
        dw2 = np.dot(self.activation1.T, delta2)
        db2 = np.sum(delta2, axis=0)

        delta1 = np.dot(delta2, self.w2.T) * (self.activation1 > 0)
        dw1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        self.w3 -= learning_rate * dw3
        self.b3 -= learning_rate * db3
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

    def train(self, X, y, learning_rate, epochs):
        for i in range(epochs):
            z1 = np.dot(X, self.w1) + self.b1
            self.activation1 = self.relu(z1)
            z2 = np.dot(self.activation1, self.w2) + self.b2
            self.activation2 = self.relu(z2)
            z3 = np.dot(self.activation2, self.w3) + self.b3
            y_hat = self.softmax(z3)
