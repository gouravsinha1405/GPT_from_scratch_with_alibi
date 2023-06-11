from blocks import *
if __name__ == "__main__":
    num_blocks, context_length, embed_size, num_heads, head_size, vocab_size, batch_size = 4, 256, 512, 8, 48, 79, 64
    model =  Decoder(num_blocks, context_length, embed_size, num_heads, head_size, vocab_size)
    model.to(device)
    x = torch.randint(0,10,(5,512))
    x = x.to(device)
    mask = torch.ones((5,512))
    mask[:, 256:] = 0
    mask = mask.to(device)
    y = model(x, mask)
    print(y.shape) #must return torch tensor of size 5,512,79