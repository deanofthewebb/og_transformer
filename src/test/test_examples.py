#%%
import torch
from layers.loop_attention import MultiHeadAttention as LoopMultiHeadAttention
import os
import sys
import pathlib

PACKAGE_PARENT = pathlib.Path.cwd().parent
sys.path.append(str(PACKAGE_PARENT))


#%%
toy_encodings = torch.Tensor([[[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]])
# shape(toy_encodings) = [B x T x D] = (1,3,4)
print(toy_encodings)

toy_MHA_layer = LoopMultiHeadAttention(d_model=4, num_heads=2)
toy_MHA, _ = toy_MHA_layer(toy_encodings, toy_encodings, toy_encodings, None)
print("Toy MHA: \n", toy_MHA)
print("Toy MHA Shape: \n", toy_MHA.shape)

#%%
from layers.residual_layer_norm import ResidualLayerNorm
toy_prev_x = torch.randn(1, 3, 4)
toy_norm_layer = ResidualLayerNorm(d_model=4)
toy_norm = toy_norm_layer(toy_encodings, toy_prev_x)

print("Toy Norm: \n", toy_norm)
print("Toy Norm shape: \n", toy_norm.shape)

#%%
from layers.pw_ffn import PWFFN
toy_PFFN_layer = PWFFN(d_model=4, d_ff=16)
toy_PWFFN = toy_PFFN_layer(toy_norm)
print("Toy PWFFN: \n", toy_PWFFN)
print("Toy PWFFN shape: \n", toy_PWFFN.shape)

#%%
from layers.encoder_layer import EncoderLayer

toy_encoder_layer = EncoderLayer(d_model=4, num_heads=2, d_ff=16)
toy_encoder_layer_outputs, toy_encoder_attn_weights = toy_encoder_layer(toy_encodings, None)
print("Toy Encoder Layer Attn Weights: \n", toy_encoder_attn_weights)
print("Toy Encoder Layer Attn Weights shape: \n", toy_encoder_attn_weights.shape)

print("Toy Encoder Layer Outputs: \n", toy_encoder_layer_outputs)
print("Toy Encoder Layer Outputs shape: \n", toy_encoder_layer_outputs.shape)


#%%
from layers.embed import Embeddings
toy_vocab = torch.LongTensor([[1, 2, 3, 4, 0, 0]])
toy_embeddings_layer = Embeddings(5, pad_idx=0, d_model=4)
toy_embeddings = toy_embeddings_layer(toy_vocab)

print("Toy Embeddings: \n", toy_embeddings)
print("Toy Embeddings shape: \n", toy_embeddings.shape)

#%%
from layers.positional_encoding import PositionalEncoding
toy_PE_layer = PositionalEncoding(d_model=4)
toy_PEs = toy_PE_layer(toy_embeddings)

print("Toy PositionalEncodings: \n", toy_PEs)
print("Toy Positional Encodings shape: \n", toy_PEs.shape)

#%%
from layers.encoder import EncoderStack
toy_encoder = EncoderStack(toy_embeddings_layer, 4, 2, 2, 16)
toy_tokenized_inputs = torch.LongTensor([[1, 2, 3, 4, 0, 0]])

toy_encoder_output, toy_encoder_attn_weights = toy_encoder(toy_tokenized_inputs)

print("Toy Encodings: \n", toy_encoder_output)
print("Toy Encoder Attention Weights: \n", toy_encoder_attn_weights)
print("Toy Encodings Shape: \n", toy_encoder_output.shape)
print("Toy Encoder Attention Weights shape: \n", toy_encoder_attn_weights.shape)

#%%
from create_mask import create_mask
toy_mask = create_mask(10)
print("Toy Mask: \n", toy_mask)
print("Toy Mask shape: \n", toy_mask.shape)

# %%
toy_scores = torch.arange(10*10).reshape(1,10,10)
print("Toy Scores: \n", toy_scores)
print("Toy Scores shape: \n", toy_mask.shape)

#%%
# masked_fill -> Tensor, condition. Grab the value where condition is met, replace the value with -1 to denote illegal values.
toy_scores = toy_scores.masked_fill(toy_mask == False, -1)
print("Toy Scores Masked: \n", toy_scores)
