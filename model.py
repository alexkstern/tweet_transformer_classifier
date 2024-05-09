import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, n_embd, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embd, num_heads, dropout=dropout,batch_first = True)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        hidden_dim=4*n_embd
        self.proj = nn.Linear(n_embd,n_embd*3)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embd)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        q,k,v = self.proj(x).chunk(3, dim=-1)
        attn_output, attn_output_weights = self.attn(q, k, v)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(self.norm1(x)))
        return self.norm2(x)

class TweetClassifier(nn.Module):
    def __init__(self,vocab_size,n_embd,padding_length,num_heads,n_layer,dropout=0.1,num_classes=2):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd) 
        self.position_embedding = nn.Embedding(num_embeddings=padding_length, embedding_dim=n_embd)
        self.norm0 = nn.LayerNorm(n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd, num_heads, dropout) for _ in range(n_layer)])
        self.fnlayernorm=nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd*2, num_classes, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x,targets):
        B,T = x.shape
        token_embedding = self.token_embedding(x)
        positions=torch.arange(T,device=x.device)
        position_embedding = self.position_embedding(positions)
        x=token_embedding+position_embedding
        x=self.norm0(x)
        x=self.blocks(x)
        x=self.fnlayernorm(x) #(B,T=282,n_embd)
        # Exclude the first token, then take the mean of the sequence
        pooled_x = x[:, 1:, :].mean(dim=1)  # Shape (B, n_embd)
        
        # Take the embedding of the first token
        first_token_embedding = x[:, 0, :]  # Shape (B, n_embd)
        
        # Concatenate pooled vector and the first token's embedding
        x = torch.cat((first_token_embedding, pooled_x), dim=1)  # Shape (B, n_embd * 2)
        
        logits = self.lm_head(x)
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = self.loss_fn(logits, targets)

        return logits, loss

    
"""
# Parameters for demonstration
vocab_size = 10000  # Number of unique tokens
n_embd = 128        # Embedding size
padding_length = 282
num_heads = 8
n_layer = 4
dropout = 0.1
num_classes = 2   

"""

