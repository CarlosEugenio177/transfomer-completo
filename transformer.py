import numpy as np
from attention import self_attention, add_and_norm, feed_forward, softmax
from decoder import cross_attention, look_ahead_mask

def encoder_block(X):

    X_att = self_attention(X)
    X_norm1 = add_and_norm(X, X_att)

    X_ffn = feed_forward(X_norm1)
    X_out = add_and_norm(X_norm1, X_ffn)

    return X_out

def encoder_stack(X, num_layers=6):

    Z = X.copy()

    for _ in range(num_layers):
        Z = encoder_block(Z)

    return Z

def decoder_block(Y, Z):
    return Y

def output_projection(Y, vocab_size):
    return Y

def run_inference():
    pass

if __name__ == "__main__":
    run_inference()