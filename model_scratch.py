import numpy as np

from multiprocessing import Pool
from functools import partial
from dnn_utils_v2 import sigmoid, relu, relu_backward, sigmoid_backward
def zero_pad(X, pad):
    """Add padding of zeros around the input.
    
    Args:
        X: Input array of shape (m, n_H, n_W, n_C)
        pad: Integer, amount of padding
        
    Returns:
        X_pad: Padded array of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    return np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant')

def conv_single_step(a_slice_prev, W, b):
    """Apply one filter on a single slice of the output activation.
    
    Args:
        a_slice_prev: Slice of input data of shape (f, f, n_C_prev)
        W: Weight parameters
        b: Bias parameters
    
    Returns:
        Z: Result of convolution
    """
    Z = np.sum(np.multiply(a_slice_prev, W)) + float(b)
    return Z

def initialize_conv_parameters(f, n_C_prev, n_C):
    """Initialize parameters for a convolutional layer.
    
    Args:
        f: Filter size (f x f)
        n_C_prev: Number of channels in prev layer 
        n_C: Number of filters
        
    Returns:
        W, b: Weight and bias parameters
    """
    # He initialization
    W = np.random.randn(f, f, n_C_prev, n_C) * np.sqrt(2.0 / (f * f * n_C_prev))
    b = np.zeros((1, 1, 1, n_C))
    return W, b

def conv_forward(A_prev, W, b, hparameters):
    """Forward propagation for a convolution layer with proper padding."""
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Proper padding to maintain dimensions
    A_prev_pad = zero_pad(A_prev, pad)
    
    # Calculate output dimensions (should maintain input dimensions with proper padding)
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    # Initialize output
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Implement convolution
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
    
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def pool_forward(A_prev, hparameters, mode="max"):
    """Forward pass for pooling layer with proper striding."""
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Fix output dimension calculation
    n_H = (n_H_prev + stride - 1) // stride  # This ensures proper rounding up
    n_W = (n_W_prev + stride - 1) // stride
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = min(vert_start + f, n_H_prev)
                    horiz_start = w * stride
                    horiz_end = min(horiz_start + f, n_W_prev)
                    
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    cache = (A_prev, hparameters)
    return A, cache

def conv_backward(dZ, cache):
    # Retrieve information from cache
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Remove ReLU backward here since it should be separate from conv
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]
        
        if pad > 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad
    
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """

    ### START CODE HERE ### (≈1 line)
    mask = x == np.max(x)
    ### END CODE HERE ###

    return mask
def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape

    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """

    ### START CODE HERE ###
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape
    # Compute the value to distribute on the matrix (≈1 line)
    average = dz / (n_H * n_W)
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones(shape) * average
    ### END CODE HERE ###

    return a
def pool_backward(dA, cache, mode="max"):
    """Backward pass for pooling layer."""
    A_prev, hparameters = cache
    
    # Get dimensions
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape
    
    # Get hyperparameters 
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Update padding calculation
    pad_h = 0 if n_H_prev % stride == 0 else (stride - (n_H_prev % stride))
    pad_w = 0 if n_W_prev % stride == 0 else (stride - (n_W_prev % stride))
    
    # Initialize gradients
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = min(vert_start + f, n_H_prev)
                    horiz_start = w * stride  
                    horiz_end = min(horiz_start + f, n_W_prev)
                    
                    if mode == "max":
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i,h,w,c]
                    
                    elif mode == "average":
                        # Improved average pooling gradient
                        da = dA[i,h,w,c]
                        shape = (vert_end - vert_start, horiz_end - horiz_start)
                        avg_mask = np.ones(shape) / (shape[0] * shape[1])
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += avg_mask * da
    
    return dA_prev

############## DENSE LAYER ##############

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)          

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    try:
        # Reshape A to 2D if needed
        if len(A.shape) > 2:
            A = A.reshape(-1, A.shape[0])
            
        # Ensure dimensions match
        if A.shape[0] != W.shape[1]:
            A = A.T
            
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        return Z, cache
        
    except ValueError as e:
        print(f"Shape error - W: {W.shape}, A: {A.shape}, b: {b.shape}")
        raise e

def linear_activation_forward(A_prev, W, b, activation):
    # Add shape validation
    if len(A_prev.shape) > 2:
        A_prev = A_prev.reshape(A_prev.shape[0], -1)
    
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
    else:
        raise ValueError(f'Activation "{activation}" not recognized')
        
    cache = (linear_cache, activation_cache)
    
    # Validate output shape
    if A.shape[0] != W.shape[0]:
        A = A.T
    
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    # Ensure input is properly shaped
    if len(A.shape) > 2:
        A = A.reshape(-1, A.shape[0])  # Changed reshape order to match dimensions
    
    # Handle hidden layers
    for l in range(L-1):
        A_prev = A 
        A, cache = linear_activation_forward(
            A_prev,
            parameters['W' + str(l+1)], 
            parameters['b' + str(l+1)], 
            "relu"
        )
        caches.append(cache)
    
    # Handle output layer
    AL, cache = linear_activation_forward(
        A,
        parameters['W' + str(L)],
        parameters['b' + str(L)],
        "softmax"
    )
    caches.append(cache)
    
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    # Use log-sum-exp trick for numerical stability
    cost = -1/m * np.sum(Y * (np.log(AL + 1e-8)))
    return np.squeeze(cost)

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dA_prev=W.T@dZ
    dW=1/m*dZ@A_prev.T
    db=1/m*np.sum(dZ,axis=1,keepdims=True)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ=relu_backward(dA,activation_cache)
    elif activation == "sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ,linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    dAL = AL - Y
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    cache = Z
    return A, cache

def softmax_backward(dA, cache):
    Z = cache
    A, _ = softmax(Z)
    dZ = A * (dA - np.sum(dA * A, axis=0, keepdims=True))
    return dZ