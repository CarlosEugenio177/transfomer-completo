import numpy as np
from attention import self_attention, add_and_norm, feed_forward, softmax
from decoder import cross_attention, look_ahead_mask

def encoder_block(X):
    return X

def encoder_stack(X, num_layers=6):
    Z = X.copy()
    return Z

def decoder_block(Y, Z):
    return Y

def output_projection(Y, vocab_size):
    return Y

def run_inference():
    pass

if __name__ == "__main__":
    run_inference()