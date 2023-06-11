from multihead_attn import *

class DecoderBlock(nn.Module):

  def __init__(self, num_heads, embed_size,context_length):
    super().__init__()
    head_size = embed_size // num_heads
    self.masked_multiheads = MultiHeadAttention(num_heads, head_size,embed_size,context_length,causal_mask=True)
    self.multiheads = MultiHeadAttention(num_heads, head_size,embed_size,context_length,causal_mask=False)
    self.feedforward = FeedForward(embed_size)
    self.ln1 = nn.LayerNorm(embed_size)
    self.ln2 = nn.LayerNorm(embed_size)
    self.ln3 = nn.LayerNorm(embed_size)
    self.dropout1 = nn.Dropout(dropout_prob)
    self.dropout2 = nn.Dropout(dropout_prob)
    self.dropout3 = nn.Dropout(dropout_prob)

  def forward(self,decoder_input,dec_mask=None):
    #adding residual connection
    decoder_input = decoder_input + self.dropout1(self.masked_multiheads(self.ln1(decoder_input),self.ln1(decoder_input),self.ln1(decoder_input), mask = dec_mask))
    # decoder_input = decoder_input + self.dropout1(self.masked_multiheads(self.ln2(decoder_input),self.ln2(encoder_output),self.ln2(encoder_output), mask = enc_mask))
    decoder_input  = decoder_input + self.dropout3(self.feedforward(self.ln3(decoder_input)))
    return decoder_input

class Decoder(nn.Module):

  def __init__(self, num_blocks, context_length, embed_size, num_heads, head_size, vocab_size):
    super().__init__()
    self.embeddings = Embedding(vocab_size,embed_size)
    # self.position_embeddings = PositionalEmbedding(context_length, embed_size)
    self.decoder_blocks = nn.Sequential(*[DecoderBlock(num_heads, embed_size,context_length) for _ in range(num_blocks)])
    self.ln1 = nn.LayerNorm(embed_size)
    self.linear_layer = nn.Linear(embed_size, vocab_size)
  
  def forward(self, dec_input,dec_mask=None):
    embed_output = self.embeddings(dec_input)
    # pos_out = self.position_embeddings(embed_output) #----these are our inputs to the block
    for block in self.decoder_blocks:
      embed_output = block(embed_output, dec_mask)
    out = self.ln1(embed_output)
    out = self.linear_layer(out)
    # out_probs = F.softmax(self.linear_layer(pos_out), dim = -1)
    return out