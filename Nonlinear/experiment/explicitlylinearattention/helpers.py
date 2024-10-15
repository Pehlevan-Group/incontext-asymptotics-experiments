import numpy as np
from data import fulldiversitysampler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Model Parameters Initialization
def initialize_parameters(length):
    # Initialize V, K, Q matrices with random values
    V = 0.1 * np.random.randn(length, length)
    K = 0.1 * np.random.randn(length, length)
    Q = 0.1 * np.random.randn(length, length)
    return V, K, Q

# Linear Attention Model for a single sample
def linear_attention(Z_sample, V, K, Q):
    # Compute (1/LENGTH) * VZ(KZ)^T(QZ)
    length = Z_sample.shape[1]
    
    # Matrix multiplication
    VZ = np.matmul(V, Z_sample)  # Shape: (SAMPLE_SIZE, LENGTH, DIM + 1)
    KZ = np.matmul(K, Z_sample)  # Shape: (SAMPLE_SIZE, LENGTH, DIM + 1)
    QZ = np.matmul(Q, Z_sample)  # Shape: (SAMPLE_SIZE, LENGTH, DIM + 1)

    # Transpose KZ along the last two axes for correct broadcasting
    KZ_T = np.transpose(KZ, (0, 2, 1))  # Shape: (SAMPLE_SIZE, DIM + 1, LENGTH)
    # print("KZT shape is", KZ_T.shape, " and should be P d+1 ell")

    # Perform batch matrix multiplication
    intermediate_result = np.matmul(VZ, KZ_T)  # Shape: (SAMPLE_SIZE, LENGTH, LENGTH)
    # print("intermediate_result shape is", intermediate_result.shape, " and should be P ell ell")

    attention_matrix = (1.0 / length) * np.matmul(intermediate_result, QZ)  # Shape: (SAMPLE_SIZE, LENGTH, DIM + 1)

    # Return the (LENGTH, DIM+1)th element (bottom-right corner element)
    return attention_matrix[:,-1, -1] 

# Loss Function (Mean Squared Error)
def compute_loss(Z, YTrue, V, K, Q):
    predictions = linear_attention(Z, V, K, Q)
    return np.mean((predictions - YTrue) ** 2)

# Gradient Descent Update Step
def sgd_update(Z, YTrue, V, K, Q, lr):
    grad_V = np.zeros_like(V); grad_K = np.zeros_like(K); grad_Q = np.zeros_like(Q)
    sample_size = Z.shape[0]
    
    for i in range(sample_size):
        dV = np.zeros_like(V); 
        Z_sample = Z[i]
        length = Z_sample.shape[0]
        
        # # Forward pass
        VZ = np.matmul(V, Z_sample); KZ = np.matmul(K, Z_sample); QZ = np.matmul(Q, Z_sample) # Shape: (LENGTH, DIM + 1)
        KZ_T = KZ.T  #(DIM + 1, LENGTH)
        intermediate_result = np.matmul(VZ, KZ_T)  # Shape: (LENGTH, LENGTH)
        attention_matrix = (1.0 / length) * np.matmul(intermediate_result, QZ)  # Shape: (LENGTH, DIM + 1)
        attention_output = attention_matrix[-1, -1]
        error = attention_output - YTrue[i]
        
        # Backward pass (gradients)
        d_attention = 2 * error / length  # Derivative of MSE loss wrt output
        
        # Gradients wrt V, K, Q
        dV[-1,:] = d_attention * ((np.matmul(Z_sample, np.matmul(KZ_T,QZ))).T)[-1,:]
        dK = d_attention * np.array([[(np.matmul(VZ, Z_sample.T)[-1,col])*(QZ[row,-1]) for col in range(length)] for row in range(length)])
        dQ = d_attention * np.array([[((np.matmul(VZ,KZ_T))[-1,row])*(Z_sample[col,-1]) for col in range(length)] for row in range(length)])
        grad_V += dV
        grad_K += dK
        grad_Q += dQ
    
    # SGD update
    V -= lr * grad_V / sample_size
    K -= lr * grad_K / sample_size
    Q -= lr * grad_Q / sample_size
    
    return V, K, Q


# # Training function
def train_model(Z, YTrue, lr=0.01, epochs=100):
    sample_size, length, dim_plus_one = Z.shape
    V, K, Q = initialize_parameters(length)
    tracking = []
    for epoch in range(epochs):
        # Compute loss
        loss = compute_loss(Z, YTrue, V, K, Q)
        tracking.append(loss)
        # print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')
        
        # Update parameters
        V, K, Q = sgd_update(Z, YTrue, V, K, Q, lr)
    
    return tracking, V, K, Q

