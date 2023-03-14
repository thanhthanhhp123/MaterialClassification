import numpy as np
import math


class NeuralNetwork():
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lr):
        # Khởi tạo trọng số ngẫu nhiên
        self.w1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((hidden_size1,))
        self.w2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((hidden_size2,))
        self.w3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((output_size,))
        self.lr = 1e-5

    def relu(self, x):
        return np.maximum(0, x)
    def softmax(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward(self, X:np.ndarray):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.w3) + self.b3
        y_hat = self.softmax(z3)
        return y_hat
    
    def calculate_loss(self, X, y):
        z1 = np.dot(X, self.w1) + self.b1
        self.activation1 = self.relu(z1)
        z2 = np.dot(self.activation1, self.w2) + self.b2
        self.activation2 = self.relu(z2)
        z3 = np.dot(self.activation2, self.w3) + self.b3
        y_hat = self.softmax(z3)
        loss = -np.mean(y * np.log(y_hat+1e-9))
        return loss

    def backward(self, X, y, y_hat):
        delta3 = y_hat - y
        dw3 = np.dot(self.activation2.T, delta3)
        db3 = np.sum(delta3, axis=0)

        delta2 = np.dot(delta3, self.w3.T) * (self.activation2 > 0)
        dw2 = np.dot(self.activation1.T, delta2)
        db2 = np.sum(delta2, axis=0)

        delta1 = np.dot(delta2, self.w2.T) * (self.activation1 > 0)
        dw1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        # Cập nhật trọng số và bias
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
    
    def fit(self, X, y, epochs = 100):
        self.losses = []
        for i in range(epochs):
            y_hat = self.forward(X)
            loss = self.calculate_loss(X, y)
            self.losses.append(loss)
            self.backward(X, y, y_hat)
            if i% math.ceil(epochs / 10) == 0:
                print("Epoch:", i, "Loss:", loss)