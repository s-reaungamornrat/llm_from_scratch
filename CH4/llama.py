import torch
import torch.nn as nn

from llm_from_scratch.CH4.blocks import RMSNorm, TransformerBlockLlama
from llm_from_scratch.CH4.utils import precompute_rope_params

class Llama3Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], dtype=cfg['dtype'])
        self.trf_blocks=nn.Sequential(*[TransformerBlockLlama(cfg, use_group_query=True) for _ in range(cfg['n_layers'])])
        self.final_norm=RMSNorm(cfg['emb_dim'], eps=1e-5)
        self.out_head=nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype'])

        cos, sin=precompute_rope_params(head_dim=cfg['emb_dim']//cfg['n_heads'], theta_base=cfg['rope_base'], context_length=cfg['context_length'],
                                        freq_config=cfg['rope_freq'])
        self.register_buffer('cos', cos, persistent=False) # persistent=False so the buffer does not get included in the model state_dict
        self.register_buffer('sin', sin, persistent=False) # and have to be recomputed
        self.cfg=cfg

    def forward(self, in_idx):
        """
        Args:
            in_idx (torch.Tensor): Input indices of shape (batch_size, seq_len/num_tokens)
        Returns:
            (torch.Tensor): Unnormalized next-token probabilities of size (batch_size, seq_len/num_tokens, vocab_size)
        """
        tok_embeds=self.tok_emb(in_idx) # (batch_size, num_tokens, emb_dim)
        x=tok_embeds

        num_tokens=x.shape[1]
        mask=torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        for block in self.trf_blocks: x=block(x, mask, self.cos, self.sin)
        x=self.final_norm(x)
        logits=self.out_head(x.to(self.cfg['dtype']))

        return logits
        
        
class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], dtype=cfg['dtype'])
        self.trf_blocks=nn.Sequential(*[TransformerBlockLlama(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm=RMSNorm(cfg['emb_dim'])
        self.out_head=nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype'])

    def forward(self, in_idx):
        # batch_size, seq_len=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        x=tok_embeds
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits
        
        