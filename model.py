"""
Script containing the Transformer architecture
"""

# importing libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import math
from utils import attention

class MultiHeadAttention(nn.Module):
    """
    Class to create the multi head attention layer for 
    encoder and decoder
    """

    def __init__(self, num_heads, emb_size, dropout=0.1):
        """
        Class constructor

        INPUT:
        num_head - (int) number of heads in multi head attention layer
        emb_size - (int) embedding size of the data
        dropout - (float) dropout percentage. Default value = 0.1
        """
        super(MultiHeadAttention, self).__init__()

        # making sure that the embedding size is divisible by the number
        # of heads
        assert emb_size % num_heads == 0

        # caching values
        self.emb_size = emb_size
        self.num_heads = num_heads

        # creating a single MLP layer for queries, keys and values
        self.q_linear = nn.Linear(emb_size, emb_size)
        self.k_linear = nn.Linear(emb_size, emb_size)
        self.v_linear = nn.Linear(emb_size, emb_size)
        # creating MLP layer for post attention
        self.post_att = nn.Linear(emb_size, emb_size)

        # creating dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        forward function for MultiHeadAttention

        INPUT:
        Q - (torch tensor) query for the transformer. Shape = (B, N, C)
        K - (torch tensor) keys for the transformer. Shape = (B, N, C)
        V - (torch tensor) values for the transformer. Shape = (B, N, C)
        mask - (torch tensor) mask for decoder multi head attention layer

        OUTPUT:
        att_output - (torch tensor) output of the multi head attention layer. Shape = (B, N, C)
        """

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # passing the Q, K, and V through 1 layer MLP
        Q, K, V = self.q_linear(Q), self.k_linear(K), self.v_linear(V)  # Shape = (B, N, C)

        # splitting Q, K and V based on num_heads
        batch_size = Q.shape[0]
        new_emb_size = self.emb_size // self.num_heads

        Q = Q.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
        K = K.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
        V = V.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)

        # permuting the dimensions of Q, K and V
        Q = Q.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
        K = K.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
        V = V.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)

        # calculating attention
        attn_output = attention(Q, K, V, mask, self.dropout)            # Shape = (B, H, N, C//H)

        # permuting the dimensions of attn_output and collapsing 
        # the num_heads dimension
        attn_output = attn_output.permute(0,2,1,3)                      # Shape = (B, N, H, C//H)
        attn_output = attn_output.reshape(batch_size, -1, self.emb_size)# Shape = (B, N, C)

        # applying linear layer to output of attention layer
        attn_output = self.post_att(attn_output)                        # Shape = (B, N, C)

        return attn_output

class EncoderLayer(nn.Module):
    """
    class for a single encoder layer
    """

    def __init__(self, emb_size, num_heads, ff_hidden_size, dropout=0.1):
        """
        class initializer

        INPUT:
        emb_size - (int) embedding size of the data
        num_heads - (int) number of heads in multi head attention layer
        ff_hidden_size - (int) size of the hidden layer for the feed forward network
        dropout - (float) dropout percentage. Default value = 0.1
        """
        super(EncoderLayer, self).__init__()

        # creating dropout layer
        self.dropout = nn.Dropout(dropout)

        # creating normalization layer for attention module
        self.norm_attn = nn.LayerNorm(emb_size)
        # creating normalization layer for feed forward layer
        self.norm_ff = nn.LayerNorm(emb_size)

        # creating object for multi head attention layer
        self.attn = MultiHeadAttention(num_heads, emb_size, dropout)

        # creating feed forward layer
        self.ff = nn.Sequential(nn.Linear(emb_size, ff_hidden_size),
                                nn.ReLU(), 
                                nn.Dropout(dropout),
                                nn.Linear(ff_hidden_size, emb_size))
    
    def forward(self, x):
        """
        forward pass through one encoder layer

        INPUT:
        x - (torch tensor) input data to the encoder layer. Shape = (B, N, C)

        OUTPUT:
        x - (torch tensor) output of the encoder layer. Shape = (B, N, C)
        """

        # sublayer 1: Input -> LayerNorm -> MultiHeadAttention -> Dropout -> ResidualAdd
        x = x + self.dropout(self.attn.forward(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x)))    # Shape = (B, N ,C)

        # sublayer 2: Input -> LayerNorm -> FFN -> Dropout -> ResidualAdd
        x = x + self.dropout(self.ff(self.norm_ff(x)))                                                      # Shape = (B, N ,C)

        return x

class Encoder(nn.Module):
    """
    class for implementing a stack of n EncoderLayers
    """

    def __init__(self, emb_size, num_heads, ff_hidden_size, n, dropout=0.1):
        """
        class initializer

        INPUT:
        emb_size - (int) embedding size of the data
        num_heads - (int) number of heads in multi head attention layer
        ff_hidden_size - (int) size of the hidden layer for the feed forward network
        n - (int) number of encoder layers 
        dropout - (float) dropout percentage. Default value = 0.1
        """
        super(Encoder, self).__init__()
        
        # creating object for 1 encoder layer
        encoder_layer_obj = EncoderLayer(emb_size, num_heads, ff_hidden_size, dropout)
        # creating a stack of n encoder layers
        self.enc_layers = nn.ModuleList([deepcopy(encoder_layer_obj) for _ in range(n)])

        # defining LayerNorm for last layer of encoder
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        """
        forward function to implement one pass through all layers of encoder

        INPUT:
        x - (torch tensor). input data. Shape = (B, N, C)

        OUTPUT:
        x - (torch tensor). output of the encoder block. Shape = (B, N, C)
        """

        for layer in self.enc_layers:
            x = layer.forward(x)               # Shape = (B, N, C)
        
        x = self.norm(x)                        # Shape = (B, N, C)

        return x

class DecoderLayer(nn.Module):
    """
    class for implementing a single decoder layer
    """

    def __init__(self, emb_size, num_heads, ff_hidden_size, dropout=0.1):
        """
        class initializer

        INPUT:
        emb_size - (int) embedding size of the data
        num_heads - (int) number of heads in multi head attention layer
        ff_hidden_size - (int) size of the hidden layer for the feed forward network
        dropout - (float) dropout percentage. Default value = 0.1
        """
        super(DecoderLayer, self).__init__()

        # creating dropout layer
        self.dropout = nn.Dropout(dropout)

        # creating normalization layer for self attention module
        self.norm_attn = nn.LayerNorm(emb_size)
        # creating normalization layer for encoder-decoder attention module
        self.norm_enc_dec = nn.LayerNorm(emb_size)
        # creating normalization layer for feed forward layer
        self.norm_ff = nn.LayerNorm(emb_size)

        # creating object for multi head self attention layer
        self.attn = MultiHeadAttention(num_heads, emb_size, dropout)
        # creating object for multi head encoder-decoder attention layer
        self.enc_dec_attn = MultiHeadAttention(num_heads, emb_size, dropout)

        # creating feed forward layer
        self.ff = nn.Sequential(nn.Linear(emb_size, ff_hidden_size),
                                nn.ReLU(), 
                                nn.Dropout(dropout),
                                nn.Linear(ff_hidden_size, emb_size))

    def forward(self, x, enc_output, source_mask, target_mask):
        """
        forward pass through one decoder layer

        INPUT:
        x - (torch tensor) input data to the decoder layer. Shape = (B, N, C)
        enc_output - (torch tensor) output of the encoder block. Shape = (B, N, C)
        source_mask - (torch tensor) mask for encoder-decoder attention layer
        target_mask - (torch tensor) mask for decoder self attention layer

        OUTPUT:
        x - (torch tensor) output of the decoder layer. Shape = (B, N ,C)
        """

        # sublayer 1: Input -> LayerNorm -> MultiHeadAttention -> Dropout -> ResidualAdd
        x = x + self.dropout(self.attn.forward(self.norm_attn(x),\
            self.norm_attn(x),self.norm_attn(x), target_mask))                          # Shape = (B, N ,C)
        
        # sublayer 2: Input -> LayerNorm -> EncoderDecoderAttention -> Dropout -> ResidualAdd
        x = x + self.dropout(self.enc_dec_attn.forward(self.norm_enc_dec(x),\
            self.norm_enc_dec(enc_output),self.norm_enc_dec(enc_output), source_mask))  # Shape = (B, N ,C)
        
        # sublayer 3: Input -> LayerNorm -> FFN -> Dropout -> ResidualAdd
        x = x + self.dropout(self.ff(self.norm_ff(x)))                                  # Shape = (B, N ,C)

        return x
        
class Decoder(nn.Module):
    """
    class for implementing stack of n decoder layers
    """

    def __init__(self, emb_size, num_heads, ff_hidden_size, n, dropout=0.1):
        """
        class initializer

        INPUT:
        emb_size - (int) embedding size of the data
        num_heads - (int) number of heads in multi head attention layer
        ff_hidden_size - (int) size of the hidden layer for the feed forward network
        n - (int) number of encoder layers 
        dropout - (float) dropout percentage. Default value = 0.1      
        """
        super(Decoder, self).__init__()

        # creating object for 1 decoder layer
        decoder_obj = DecoderLayer(emb_size, num_heads, ff_hidden_size, dropout)
        # creating stack of n decoder layers
        self.dec_layers = nn.ModuleList([deepcopy(decoder_obj) for _ in range(n)])

        # defining LayerNorm for decoder end
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x, enc_output, source_mask, target_mask):
        """
        x - (torch tensor) input data to the decoder block. Shape = (B, N, C)
        enc_output - (torch tensor) output of the encoder block. Shape = (B, N, C)
        source_mask - (torch tensor) mask for encoder-decoder attention layer
        target_mask - (torch tensor) mask for decoder self attention layer

        OUTPUT:
        x - (torch tensor) output of the decoder layer. Shape = (B, N ,C)
        """

        for layer in self.dec_layers:
            x = layer.forward(x, enc_output, source_mask, target_mask)      # Shape = (B, N, C)
        
        x = self.norm(x)                                                    # Shape = (B, N, C)

        return x

class PositionalEncoding(nn.Module):
    """
    class to implement positional encoding for encoder and decoder input data
    """

    def __init__(self, emb_size, dropout=0.1, max_len=5000):
        """
        class initializer

        INPUT:
        emb_size - (int) size of the embedding
        dropout - (float) dropout percentage. Default value = 0.1
        max_len - (int) max positional length. Default value = 5000
        """
        super(PositionalEncoding, self).__init__()

        # defining the dropout layer
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        forward pass to generate positional embeddings

        INPUT:
        x - (torch tensor) embedded data. Shape = (B, N, C) 

        OUTPUT:
        x - (torch tensor) positional embedded data. Shape = (B, N, C) 
        """

        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = self.dropout(x)

        return x

class Embeddings(nn.Module):
    """
    class to generate the embeddings for encoder and decoder input data
    """

    def __init__(self, input_size, emb_size):
        """
        class initializer

        INPUT:
        input_size - (int) size of the input data
        emb_size - (int) size of the embedding
        """
        super(Embeddings, self).__init__()

        # caching values
        self.emb_size = emb_size

        # creating liner layer for embedding input data
        self.linear_embd = nn.Linear(input_size, emb_size)

        # creating object for positional encoding
        self.pos_encoding = PositionalEncoding(emb_size, dropout=0.1, max_len=5000)
    
    def forward(self, x):
        """
        forward pass to generate input embeddings

        INPUT:
        x - (torch tensor) input data. Shape = (B, N, input_dimension)

        OUTPUT:
        x - (torch tensor) embedded data. Shape = (B, N, C)
        """

        # creating embeddings for input data
        x = self.linear_embd(x) * math.sqrt(self.emb_size)     # Shape = (B, N, C)
        # incorporating positional embeddings
        x = self.pos_encoding.forward(x)

        return x

class OutputGenerator(nn.Module):
    """
    class to generate the output embeddings from the transformer's output
    """

    def __init__(self, emb_size, output_size):
        """
        class initializer

        INPUT:
        output_size - (int) size of the output data
        emb_size - (int) size of the embedding
        """
        super(OutputGenerator, self).__init__()

        # creating liner layer for embedding input data
        self.output_gen = nn.Linear(emb_size, output_size)
    
    def forward(self, x):
        """
        forward pass to generate the output data

        INPUT:
        x - (torch tensor) input data from transformer. Shape = (B, N, output_dimension)

        OUTPUT:
        x - (torch tensor) output data. Shape = (B, N, output_size)
        """

        x = self.output_gen(x)     # Shape = (B, N, output_size) 

        return x

class TFModel(nn.Module):
    """
    class to generate the complete transformer architecture
    """

    def __init__(self, encoder_ip_size, decoder_ip_size, model_op_size, emb_size, \
                num_heads, ff_hidden_size, n, dropout=0.1):
        """
        class initializer

        INPUT:
        encoder_ip_size - (int) dimension of the encoder input
        decoder_ip_size - (int) dimension of the decoder input
        model_op_size - (int) dimension of model's output
        emb_size - (int) data embedding size for encoder and decoder
        num_heads - (int) number of heads in multi head attention layer
        ff_hidden_size - (int) size of the hidden layer for the feed forward network
        n - (int) number of encoder layers 
        dropout - (float) dropout percentage. Default value = 0.1
        """
        super(TFModel, self).__init__()

        # creating embeddings for encoder input
        self.encoder_embedding = Embeddings(encoder_ip_size, emb_size)
        # creating embeddings for decoder input
        self.decoder_embeddings= Embeddings(decoder_ip_size, emb_size)
        
        # creating encoder block
        self.encoder_block = Encoder(emb_size, num_heads, ff_hidden_size, n, dropout)
        # creating decoder block
        self.decoder_block = Decoder(emb_size, num_heads, ff_hidden_size, n, dropout)

        # creating output generator
        self.output_gen = OutputGenerator(emb_size, model_op_size)
    
    def forward(self, enc_input, dec_input, dec_source_mask, dec_target_mask):
        """
        forward pass for the transformer model

        INPUT:
        enc_input - (torch tensot) input data to the encoder block. Shape = (B, N, encoder_ip_size)
        dec_input - (torch tensor) input data to the decoder block. Shape = (B, N, decoder_ip_size)
        enc_output - (torch tensor) output of the encoder block. Shape = (B, N, emb_size)
        source_mask - (torch tensor) mask for encoder-decoder attention layer
        target_mask - (torch tensor) mask for decoder self attention layer

        OUTPUT:
        model_output - (torch tensor) output of the model. Shape = (B, N, model_op_size)
        """

        enc_embed = self.encoder_embedding.forward(enc_input)
        encoder_output = self.encoder_block.forward(enc_embed)

        dec_embed = self.decoder_embeddings.forward(dec_input)
        decoder_output = self.decoder_block.forward(dec_embed, encoder_output, dec_source_mask, dec_target_mask)

        model_output = self.output_gen.forward(decoder_output)

        return model_output
