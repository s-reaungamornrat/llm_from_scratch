import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from llm_from_scratch.CH4.utils import precompute_rope_params, compute_rope 

class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):

        super().__init__()
        assert (d_out%num_heads==0), f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"

        self.d_out=d_out
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads # reduce the projection dim to match the desired output dim
        self.W_query=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj=nn.Linear(d_out, d_out) # use a linear layer to combine head outputs
        self.dropout=nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in=x.shape
        keys=self.W_key(x) # (b, num_tokens, d_out)
        queries=self.W_query(x)
        values=self.W_value(x)

        # We implicitly split the matrix by adding num_heads dimension
        keys=keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values=values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries=queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # swap between num_tokens and num_heads
        keys=keys.transpose(1,2) # (b, num_heads, num_tokens, head_dim)
        queries=queries.transpose(1,2)
        values=values.transpose(1,2)

        # Compute the dot product for each head
        # (b, num_heads, num_tokens, head_dim)@(b, num_heads, head_dim, num_tokens) = (b, num_heads, num_tokens, num_tokens)
        attn_scores=queries @ keys.transpose(-2,-1)
        mask_bool=self.mask.bool()[:num_tokens, :num_tokens] # truncate mask to the number of tokens
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights=self.dropout(attn_weights)

        # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)=(b, num_heads, num_tokens, head_dim)
        context_vec=(attn_weights@values).transpose(1,2) # -> (b, num_tokens, num_heads, head_dim)
        context_vec=context_vec.contiguous().view(b, num_tokens, self.d_out) # combine num_heads and head_dim to d_out
        context_vec=self.out_proj(context_vec)
        return context_vec

class MultiheadAttentionLlama(nn.Module):

    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        super().__init__()
        assert d_out%num_heads==0, f"d_out {d_out} must be divisible by n_heads {n_heads}"

        self.d_out=d_out
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads # reduce the projection dim to match desired output dim
        # set bias=False and dtype for all linear layers below
        self.W_query=nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key=nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_value=nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj=nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        cos, sin=precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        b, num_tokens, d_in=x.shape

        keys=self.W_key(x) # (b, num_tokens, d_out)
        queries=self.W_query(x)
        values=self.W_value(x)

        # we split the matrices by adding a `num_heads` dimension
        keys=keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values=values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries=queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose from (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys=keys.transpose(1,2)
        queries=queries.transpose(1,2)
        values=values.transpose(1,2)

        keys=compute_rope(keys, self.cos, self.sin)
        queries=compute_rope(queries, self.cos, self.sin)

        # compute scaled dot-product attention (self-attention) with a casual mask
        # (b, num_heads, num_tokens, head_dim)@(b, num_heads, head_dim, num_tokens) = (b, num_heads, num_tokens, num_tokens)
        attn_scores=queries@keys.transpose(2,3) # dot product for each head
        
        # original mask truncated to the number of tokens and converted to boolean
        mask_bool=self.mask.bool()[:num_tokens, :num_tokens]

        # use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) # (b, num_heads, num_tokens, num_tokens)

        # (b, num_heads, num_tokens, num_tokens)@(b, num_heads, num_tokens, head_dim)=(b, num_heads, num_tokens, head_dims)
        # ->(b, num_tokens, num_heads, head_dim)
        context_vec=(attn_weights@values).transpose(1,2)
        # combine heads where d_out=num_heads*head_dim
        context_vec=context_vec.reshape(b, num_tokens, self.d_out)
        context_vec=self.out_proj(context_vec)

        return context_vec


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        """
        To make GroupedQueryAttention equivalent to MultiheadAttention, one can set num_kv_groups to num_heads
        """
        super().__init__()
        assert d_out % num_heads==0, f"d_out {d_out} must be divisible by num_heads {num_heads}"
        assert num_heads % num_kv_groups==0, f"num_heads {num_heads} must be divisible by num_kv_groups {num_kv_groups}"

        self.d_out=d_out
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads

        self.W_key=nn.Linear(d_in, num_kv_groups*self.head_dim, bias=False, dtype=dtype)
        self.W_value=nn.Linear(d_in, num_kv_groups*self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups=num_kv_groups
        self.group_size=num_heads//num_kv_groups

        self.W_query=nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj=nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask=None, cos=None, sin=None):
        """
        Args:
            x (torch.Tensor): Input embeddings of shape (b, num_tokens, d_in) 
            mask (torch.Tensor|None): Casual mask of shape (num_tokens, num_tokens)
            cos (torch.Tensor): cosine of shape (context_length, head_dim), where context_length>=num_tokens
            sin (torch.Tensor): sine of shape (context_length, head_dim), where context_length>=num_tokens
        Returns:
            (torch.Tensor): Embedding of shape (b, num_tokens, d_out)
        """
        
        b, num_tokens, d_in=x.shape

        queries=self.W_query(x) # (b, num_tokens, d_out)
        keys=self.W_key(x) # (b, num_tokens, num_kv_groups*head_dim)
        values=self.W_value(x) # (b, num_tokens, num_kv_groups*head_dim)

        # reshape queries, keys, and values
        queries=queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys=keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values=values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # transpose keys, values, and queries
        keys=keys.transpose(1,2) # (b, num_kv_groups, num_tokens, head_dim)
        values=values.transpose(1,2) # (b, num_kv_groups, num_tokens, head_dim)
        queries=queries.transpose(1,2) # (b, num_heads, num_tokens, head_dim)

        # apply RoPE
        if cos is not None:
            keys=compute_rope(keys, cos, sin)
            queries=compute_rope(queries, cos, sin)

        # expand keys and values to match the number of heads (b, num_heads, num_tokens, head_dim)
        keys=keys.repeat_interleave(self.group_size, dim=1) # (b, num_kv_groups, num_tokens, head_dim) -> (b, num_heads, num_tokens, head_dim)
        values=values.repeat_interleave(self.group_size, dim=1) # (b, num_heads, num_tokens, head_dim)
        # for example, before repeat_interleave along dim=1 (query groups)
        # [K1, K2]
        # after repeat_interleave (each query group is repeated group_size times)
        # [K1, K1, K2, K2]
        # if we use regular repeat instead of repeat_interleave, we'd get
        # [K1, K2, K1, K2]
        
        # compute scaled dot-product attention (aka self-attention) with a causal mask
        # (b, num_heads, num_tokens, head_dim)@(b, num_heads, head_dim, num_tokens)=(b, num_heads, num_tokens, num_tokens)
        attn_scores=queries@keys.transpose(2,3) 

        # create mask on the fly
        if mask is None: mask=torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        # use the mask to fill attention scores
        attn_scores.masked_fill_(mask, -torch.inf)

        assert keys.shape[-1]==self.head_dim
        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)

        # [(b, num_heads, num_tokens, num_tokens)@(b, num_heads, num_tokens, head_dim)].transpose(1,2)=
        # (b, num_heads, num_tokens, head_dim).transpose(1,2)=(b, num_tokens, num_heads, head_dim)
        context_vec=(attn_weights@values).transpose(1,2) 

        # combine heads, where self.d_out=self.num_heads*self.head_dim
        context_vec=context_vec.reshape(b, num_tokens, self.d_out)
        context_vec=self.out_proj(context_vec) 
        
        return context_vec
        

class LayerNorm(nn.Module):

    """
    Apply standard normalization to the feature dimension (assuming the last dimension) of input tensors
    """
    def __init__(self, emb_dim):
        """
        Args:
            emb_dim (int): Feature dimension
        """
        super().__init__()
        self.eps=1e-5
        # Trainable parameters that the model automatically adjusts during training if it is determined that doing so would improve
        # the model's performance
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.shift=nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of size (...,emb_dim)
        """
        mean=x.mean(dim=-1, keepdim=True)
        var=x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x=(x-mean)/(var+self.eps).sqrt()
        return self.scale*normalized_x+self.shift

class RMSNorm(nn.Module):
    """ Similar to torch.nn.RMSNorm """
    
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps=eps
        self.emb_dim=emb_dim
        self.weight=nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of size (...,emb_dim)
        """
        means=x.pow(2.).mean(dim=-1, keepdim=True)
        x_normed=x*torch.rsqrt(means+self.eps) # recipocal of square root
        return x_normed*self.weight

class GELU(nn.Module):

    def __init__(self): super().__init__()
    def forward(self, x):
        return 0.5 * x * (1.+torch.tanh(
                                     torch.sqrt(torch.tensor(2./torch.pi)) * 
                                     (x + 0.044715 * torch.pow(x, 3.))
                                   )
                     )

class SiLU(nn.Module):
    """ Similar to torch.nn.functional.silu """
    def __init__(self):
        super(SiLU, self).__init__()
    def forward(self, x): return x*torch.sigmoid(x)

class FeedForward(nn.Module):
    """ Feedforward for GPT 2"""
    def __init__(self, emb_dim):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            GELU(),
            nn.Linear(4*emb_dim, emb_dim)
        )
    def forward(self, x): 
        """
        Args:
            x (torch.Tensor): Input tensor of size (..., emb_dim)
        """
        return self.layers(x)


class FeedForwardLlama(nn.Module):
    """Feedforward for llama using Gates Linear Unit (GLU) of SiLU called SwiGLU
    see https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb
    Args:
        dtype (torch.dtype): Allow loading model directly in lower precision formats
    """
    def __init__(self, emb_dim, hidden_dim, dtype):
        super().__init__()
        self.fc1=nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False) # Llama does not use any bias unit
        self.fc2=nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3=nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)
        self.silu=SiLU()

    def forward(self, x):
        x_fc1=self.fc1(x)
        x_fc2=self.fc2(x)
        x=self.silu(x_fc1)*x_fc2
        return self.fc3(x)
        
class TransformerBlock(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.att=MultiHeadAttention(d_in=cfg['emb_dim'], d_out=cfg['emb_dim'], context_length=cfg['context_length'],
                                        num_heads=cfg['n_heads'], dropout=cfg['drop_rate'], qkv_bias=cfg['qkv_bias'])
        self.ff=FeedForward(emb_dim=cfg['emb_dim'])
        self.norm1=LayerNorm(emb_dim=cfg['emb_dim'])
        self.norm2=LayerNorm(emb_dim=cfg['emb_dim'])
        self.drop_shortcut=nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of size (..., emb_dim), typically (batch_size, num_tokens, emb_dim) where 
                num_tokens is sometimes referred to as seq_len or context_length
        """
        shortcut=x # shortcut connection for attention block
        x=self.norm1(x) # 
        x=self.att(x)
        x=self.drop_shortcut(x) 
        x=x+shortcut # add the original input back

        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x=x+shortcut # add the original input back
        return x

class TransformerBlockLlama(nn.Module):
    def __init__(self, cfg, use_group_query=False):
        super().__init__()
        self.use_group_query=use_group_query
        if not self.use_group_query:
            self.att=MultiheadAttentionLlama(d_in=cfg['emb_dim'], d_out=cfg['emb_dim'], context_length=cfg['context_length'],
                                             num_heads=cfg['n_heads'], dtype=cfg['dtype'])
        else:
            self.att=GroupedQueryAttention(d_in=cfg['emb_dim'], d_out=cfg['emb_dim'], num_heads=cfg['n_heads'], num_kv_groups=cfg['n_kv_groups'],
                                           dtype=cfg['dtype'])
        self.ff=FeedForwardLlama(emb_dim=cfg['emb_dim'], hidden_dim=cfg['hidden_dim'], dtype=cfg['dtype'])
        self.norm1=RMSNorm(cfg['emb_dim'], eps=1e-5)
        self.norm2=RMSNorm(cfg['emb_dim'], eps=1e-5)
        
    def forward(self, x, mask=None, cos=None, sin=None):
        """
        Args:
            x (torch.Tensor): Input tensor of size (..., emb_dim), typically (batch_size, num_tokens, emb_dim) where 
                num_tokens is sometimes referred to as seq_len or context_length
        """
        # short cut connection for attention block
        shortcut=x
        x=self.norm1(x)
        if self.use_group_query: x=self.att(x.to(torch.bfloat16), mask, cos, sin) # (batch_size, num_tokens, emb_dim)
        else: x=self.att(x)
        x=x+shortcut # add original input back

        # shortcut for feed-forward block
        shortcut=x
        x=self.norm2(x)
        x=self.ff(x.to(torch.bfloat16)) if self.use_group_query else self.ff(x) 
        x=x+shortcut # adding original input back

        return x
        
if __name__=="__main__":

    torch.manual_seed(123)

    # MultiHeadAttention
    inputs=torch.tensor( # 6 tokens long sentence, each token is 3D embedding
        [[0.43,0.15,0.89],
         [0.55,0.87,0.66],
         [0.57,0.85,0.64],
         [0.22,0.58,0.33],
         [0.77,0.25,0.10],
         [0.05,0.80,0.55]]
    )
    batch=torch.stack((inputs, inputs), dim=0)
    print(f"{batch.shape=}, {batch.dtype=}") # 2 input texts with 6 token each, each token is 3D embeddings
    batch_size, context_length, d_in=batch.shape
    d_out=2
    mha=MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vec=mha(batch)
    print(f"{context_vec.shape=}\n{context_vec}")


    # LayerNorm
    batch_example=torch.rand(2,5)
    layer=nn.Sequential(nn.Linear(5,6), nn.ReLU())
    out=layer(batch_example)
    
    ln=LayerNorm(emb_dim=6)
    out_ln=ln(out)
    print(f"{out.shape=}\n{out}")
    print(f"{out_ln.shape=}\n{out_ln}")
    mean=out_ln.mean(dim=-1, keepdim=True)
    var=out_ln.var(dim=-1, keepdim=True, unbiased=False)
    print(f"{mean=}\n{var=}") # 0 mean and 1 variance--> unit variance

    # GELU
    x=torch.linspace(-3, 3, 100)
    y_gelu, y_relu=GELU()(x), nn.ReLU()(x)
    fig, axes=plt.subplots(1,2,figsize=(8,3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu],['GELU', 'ReLU'])):
        axes[i].plot(x, y)
        axes[i].set_title(f"{label} activation")
        axes[i].set_xlabel('x')
        axes[i].set_ylabel(f"{label}(x)")
        axes[i].grid(True)
    plt.tight_layout()

    # Feedforward
    ffn=FeedForward(emb_dim=768)
    x=torch.rand(2,3,768)
    out=ffn(x)
    print(f"{out.shape=}, {out.dtype=}")