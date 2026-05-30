import torch
import torch.nn as nn

from llm_from_scratch.CH5.qwen35.block import GroupedQueryAttention, apply_rope, compute_rope_params, RMSNorm, FeedForward
from llm_from_scratch.CH5.qwen35.module import Qwen3_5GateDeltaNet


# mapping for the different naming convention in huggingface transformer
class _Qwen3_5ConfigAdapter:
    def __init__(self, cfg):
        self.hidden_size=cfg["emb_dim"]
        self.linear_num_value_heads=cfg["linear_num_value_heads"]
        self.linear_num_key_heads=cfg['linear_num_key_heads']
        self.linear_key_head_dim=cfg['linear_key_head_dim']
        self.linear_value_head_dim=cfg['linear_value_head_dim']
        self.linear_conv_kernel_dim=cfg['linear_conv_kernel_dim']
        self.hidden_act="silu"
        self.rms_norm_eps=cfg.get('rms_norm_eps', 1e-6)
        self.dtype=cfg.get("dtype", None)

class TransformerBlock(nn.Module):

    def __init__(self, cfg, layer_type, layer_idx):
        super().__init__()
        self.layer_type=layer_type
        if layer_type=="full_attention":
            self.token_mixer=GroupedQueryAttention(d_in=cfg['emb_dim'], num_heads=cfg['n_heads'], head_dim=cfg['head_dim'], 
                                                   num_kv_groups=cfg['n_kv_groups'], qk_norm=cfg['qk_norm'], dtype=cfg['dtype'])
        elif layer_type=="linear_attention":
            self.token_mixer=Qwen3_5GateDeltaNet(_Qwen3_5ConfigAdapter(cfg), layer_idx)
        else: raise ValueError(f"Unsupported layer type: {layer_type}")

        self.ff=FeedForward(emb_dim=cfg['emb_dim'], hidden_dim=cfg['hidden_dim'], dtype=cfg['dtype'])
        self.norm1=RMSNorm(cfg['emb_dim'], eps=cfg.get('rms_norm_eps', 1e-6))
        self.norm2=RMSNorm(cfg['emb_dim'], eps=cfg.get('rms_norm_eps', 1e-6))

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, linear_cache=None, cache_position=None):
        """
        Args:
            x (torch.Tensor): Input embedding of shape (batch_size, seq_len, emb_dim), often of type bfloat16. If cache is used,seq_len=1
            mask (torch.Tensor): Boolean causal mask of shape (1,1, seq_len, seq_len). If cache is used,seq_len=1
            cos (torch.Tensor): Cosine tensors of size (context_length, rotary_dim)
            sin (torch.Tensor): Sine tensors of size (context_length, rotary_dim)
            linear_cache (Qwen3_5LinearAttentionCache)
            cache_position (torch.Tensor): Token position from start to end, e.g., torch.arange(pos_start, pos_end). If cache is used (inference),
                this will often appear like torch.tensor([start_pos]) with only 1 number in it
        Returns:
            (torch.Tensor): Output embedding of shape (batch_size, seq_len, emb_dim) similar to the input
            (tuple[torch.Tensor]): Next cache, each of shape (b, num_kv_group, seq_len, head_dim) from full attention module
        """
        shortcut=x
        x=self.norm1(x)
        if self.layer_type=="full_attention":
            # return x of shape (batch_size, seq_len, d_out=n_heads*head_dim) which is equal to the size returns from linear_attention
            x, next_cache=self.token_mixer(x, mask, cos, sin, start_pos=start_pos, cache=cache)
        else:
            # return x of shape (batch_size, seq_len, emb_dim/hidden_size) which is equal to the size returns from full attention
            x=self.token_mixer(x, cache_params=linear_cache, cache_position=cache_position)
            next_cache=None
        x=x+shortcut
        
        shortcut=x
        x=self.norm2(x) # (batch_size, seq_len, emb_dim/hidden_size)
        x=self.ff(x) # (batch_size, seq_len, emb_dim/hidden_size)
        x=x+shortcut
        
        return x, next_cache

class Qwen3_5Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], dtype=cfg['dtype'])
        layer_types=cfg.get("layer_types", ['full_attention']*cfg['n_layers'])
        if len(layer_types) != cfg['n_layers']: raise ValueError(f"{len(layer_types)=} must equal {n_layers=}")
        self.trf_blocks=nn.ModuleList(
            [TransformerBlock(cfg, layer_type, idx) for idx, layer_type in enumerate(layer_types)]
        )
        self.final_norm=RMSNorm(cfg['emb_dim'], eps=cfg.get("rms_norm_eps", 1e-6))
        self.out_head=nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype'])

        head_dim=cfg['emb_dim']//cfg['n_heads'] if cfg['head_dim'] is None else cfg['head_dim']
        cos, sin=compute_rope_params(head_dim=head_dim, theta_base=cfg['rope_base'], context_length=cfg['context_length'],
                                     partial_rotary_factor=cfg.get("partial_rotary_factor", 1.), dtype=torch.float32)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        self.cfg=cfg
        self.current_pos=0

    def create_mask(self, cur_len, device, pos_start=0, pos_end=None):
        """
        Args:
            cur_len (int): Number of tokens for the current sequence
            pos_start (int): Index location of tokens considered the start of the sequence of interest, default is 0
            pos_end (int): Index location of tokens considered one behind the end of the sequence of interest, default is num_tokens/seq_len
        Returns:
            (torch.Tensor): Upper triangles (with 0 elsewhere including the diagonal), truncated to the shape (1,1,pos_end-pos_start, pos_end)
                if pos_start!=0
        """
        if pos_end is None: pos_end=cur_len

        ones=torch.ones((pos_end, pos_end), device=device, dtype=torch.bool) # (seq_len, seq_len)
        mask_full=torch.triu(ones, diagonal=1) # set all lower triangle including the diagonal elements to 0
        row_slice=slice(pos_start, pos_end)
        # cut/select from pos_start row to pos_end row instead of row 0-th to row pos_end-th, so its dimension is (1,1,pos_end-pos_start, pos_end)
        mask=mask_full[row_slice, :pos_end][None,None] 
        return mask

    def forward(self, in_idx, cache=None, verbose=False):
        """
        Args:
            in_idx (torch.Tensor): Input token indices of shape (batch_size, num_tokens/seq_len) of type long/int64. If KVCache is used,
                the token indices will have shape (batch_size, 1), i.e., seq_len/num_tokens=1
            cache (KVCache): A class object to store the hidden states of previous tokens
        Returns:
            logits (torch.Tensor): Output of shape (batch_size, seq_len, vocab_size)
        """        
        x=self.tok_emb(in_idx) # (batch_size, seq_len, emb_dim) without cache, and (batch_size, 1, emb_dim) with cache

        num_tokens=x.shape[1]
        if cache is not None:
            pos_start=self.current_pos # first round with cache, this will be seq_len; next round, will be seq_len+1
            pos_end=pos_start+num_tokens # first round with cache, seq_len+1; next round, will be seq_len+2
            self.current_pos=pos_end
            # (1,1,num_tokens, pos_start+num_tokens)
            mask=self.create_mask(cur_len=num_tokens, device=x.device, pos_start=pos_start, pos_end=pos_end) 
            cache_position=torch.arange(pos_start, pos_end, device=x.device, dtype=torch.long)
        else:
            pos_start=0
            mask=self.create_mask(cur_len=num_tokens, device=x.device, pos_start=0, pos_end=num_tokens) #(1,1,num_tokens, num_tokens)
            cache_position=None
            
        for i, block in enumerate(self.trf_blocks):
            blk_cache=cache.get(i) if cache is not None else None
            # output and input x is of size (batch_size, seq_len, emb_dim)
            # new_blk_cache is a tuple of 2 tensors, each is of size (b, num_kv_group, seq_len, head_dim), only estimated in full attention mode
            x, new_blk_cache=block(x, mask=mask, cos=self.cos, sin=self.sin, start_pos=pos_start, cache=blk_cache, 
                                   linear_cache=cache.linear_cache if cache is not None else None, 
                                   cache_position=cache_position)
            
            if cache is not None and new_blk_cache is not None: cache.update(i, new_blk_cache)
        if cache is not None: cache.linear_cache.has_previous_state=True

        x=self.final_norm(x) # (batch_size, seq_len, emb_dim)
        logits=self.out_head(x.to(self.cfg['dtype'])) # (batch_size, seq_len, vocab_size)

        return logits

    def reset_kv_cache(self): self.current_pos=0

class Qwen3_5LinearAttentionCache:
    def __init__(self, n_layers):
        self.conv_states=[None]*n_layers
        self.recurrent_states=[None]*n_layers
        self.has_previous_state=False
    def reset(self):
        for i in range(len(self.conv_states)):
            self.conv_states[i]=None
            self.recurrent_states[i]=None
        self.has_previous_state=False

class KVCache:
    """
    Store the hidden states of previous tokens so the model doesn't have to re-calculate them every time.
    """
    def __init__(self, n_layers):
        self.cache=[None]*n_layers
        self.linear_cache=Qwen3_5LinearAttentionCache(n_layers)
    def get(self, layer_idx): return self.cache[layer_idx]
    def update(self, layer_idx, value): self.cache[layer_idx]=value
    def get_all(self): return self.cache
    def reset(self):
        for i in range(len(self.cache)): self.cache[i]=None
        self.linear_cache.reset()
    
    