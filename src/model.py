import numpy as np
class NeuralNetwork:
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
    
    def forward():
        for epoch in range(max_epochs):
            total_loss = 0
            
            for i in range(len(x)):
                z_in = x[i] @ w + b0
                z_out = relu(z_in)
                y_in = z_out @ v + b1
                y = sigmoid(y_in)

                loss = BCE(t[i], y)
                total_loss += loss

                w, b0, v, b1 = backpropagation(x[i], y, t[i], z_in, z_out, w, b0, v, b1, lr)