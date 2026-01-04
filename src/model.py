import numpy as np
class NeuralNetwork:
    def relu(x):
        return np.maximum(0, x)

    def relu_deriv(x):
        return (x > 0).astype(float)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(x):
        return sigmoid(x) * (1-sigmoid(x))
    
    def BCE(t, y):
        return - (t * np.log(y))

    def BCE_deriv(t, y): # only when sigmoid is used
        return y - t

    def forward(max_epochs, x, w, b0, relu, v, b1, sigmoid, BCE, t):
        for epoch in range(max_epochs):
            total_loss = 0
            
            for i in range(len(x)):
                z_in = x[i] @ w + b0
                z_out = relu(z_in)
                y_in = z_out @ v + b1
                y = sigmoid(y_in)

                loss = BCE(t[i], y)
                total_loss += loss

        return y
                
    def backpropagation(self, x_i, y, t_i, z_in, z_out, w, b0, v, b1, lr):
        
        delta_out = BCE_deriv(t_i, y)
        
        dv = z_out.reshape(-1, 1) * delta_out
        db1 = delta_out

        delta_hidden = (v.flatten() * delta_out) * relu_deriv(z_in)
        
        dw = np.outer(x_i, delta_hidden)
        db0 = delta_hidden
            
        v  -= lr * dv
        b1 -= lr * db1
        w  -= lr * dw
        b0 -= lr * db0

        return w, b0, v, b1