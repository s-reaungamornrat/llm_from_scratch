import torch
import torch.nn as nn

from llm_from_scratch.CH4.blocks import LayerNorm, TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb=nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb=nn.Dropout(cfg['drop_rate'])

        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        # standardizing the transformer outputs to stabilize the learning process
        self.final_norm=LayerNorm(cfg['emb_dim'])
        self.out_head=nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        """
        Args:
            in_idx (torch.Tensor): Input token indices of size (batch_size, seq_len/num_tokens)
        Returns:
            (torch.Tensor): Unnormalized next-token probabilities of size (batch_size, seq_len/num_tokens, vocab_size)
        """
        batch_size, seq_len=in_idx.shape
        tok_embs=self.tok_emb(in_idx)
        pos_embs=self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x=tok_embs+pos_embs
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        # standardizing the transformer outputs to stabilize the learning process
        x=self.final_norm(x)  # (batch_size, seq_len, emb_dim)
        # project the transformer outputs to the vocabulary space of tokenizer ro generate logits for each token in the vocabulary
        logits=self.out_head(x)  # (batch_size, seq_len, vocab_size)
        return logits # next token's unnormalized probabilities


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Args:
        idx (torch.Tensor): Current context indices of size (batch, n_tokens)
        max_new_tokens (int): Maximum number of new tokens to be generated
        
    """
    for _ in range(max_new_tokens):
        # crop current context if it exceeds the supported context size, e.g., if LLM supports only 5 tokens, 
        # and the context size is 10, then only the last 5 tokens are used as context
        idx_cond=idx[:, -context_size:]
        with torch.no_grad(): logits=model(idx_cond) # (batch, n_tokens, vocab_size)

        # focus only on the last time step so (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits=logits[:,-1] # (batch, vocab_size)
        probs=torch.softmax(logits, dim=-1) # (batch, vocab_size)
        idx_next=torch.argmax(probs, dim=-1, keepdim=True) # (batch, 1)
        idx=torch.cat((idx, idx_next), dim=1) # (batch, n_tokens+1,2,3,..max_new_tokens)
    return idx