import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, emb_dim, hidden_dim, dtype):
        super().__init__()
        self.fc1=nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2=nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3=nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)

    def forward(self, x):
        x_fc1=self.fc1(x)
        x_fc2=self.fc2(x)
        x=torch.nn.functional.silu(x_fc1)*x_fc2
        return self.fc3(x)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps=eps
        # Qwen 3.5 uses (1+weight) scaling with zero init
        self.weight=nn.Parameter(torch.zeros(emb_dim))

    def _norm(self,x): 
        """
        Compute normalization of input by dividing the input by the root-mean-square of x 
        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim)
        """
        return x*torch.rsqrt(x.pow(2.).mean(dim=-1, keepdim=True)+self.eps)

    def forward(self, x):
        x_norm=self._norm(x.float())
        x_norm=x_norm*(1.+self.weight.float()) # itself + scaled of itself
        return x_norm.to(dtype=x.dtype)


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, partial_rotary_factor=1., dtype=torch.float32):
    """
    Args:
        head_dim (int): Dimensionality of a single attention head (e.g., 64, 128)
        theta_base (int): Base frequency constant, controling how quickly the rotation frequencies decay across the hidden dimensions
        context_length (int): Maximum sequence length (number of tokens) the model can process in a single forward pass
        partial_rotary_factor (float): A percentage between 0. and 1. for rotating only a fraction of the hidden dimensions to save compute 
            (leaving the rest unrotated, e.g., GPT-NeoX or Phi)
        dtype (torch.dtype): Data type used to perform the calculation
    Returns:
        (tuple[torch.Tensor]): Tensors of cosine and sine, each of size (context_length, rotary_dim)
    """
    # RoPe works by grouping features into pairs and rotating them like coordinates on a 2D graph (x,y). Thus, dimension must be even
    assert head_dim%2==0, "embedding dimension must be even"

    # Partial rotation, e.g., head_dim=128 and partial_rotary_factor=0.5, only 64 dimensions will get positional embeddings
    rotary_dim=int(head_dim*partial_rotary_factor)
    # Make sure that rotary_dim is even, by subtracting the remainder and ensure that rotary_dim is at least 2 (minimum size required to perform
    # 2D rotation)
    rotary_dim=max(2, rotary_dim-(rotary_dim%2))

    # A tensor for unique rotational frequency for each pair of dimensions theta_i=theta_base ** (-2i/d) where d is rotary_dim
    # Below, we ensure that the tensor is halft the length of rotary_dim since we only need one frequency per pair
    # Inversion makes early dimensions rotate repidly and later dimensions rotate incredibly slowly
    inv_freq=1./(
        theta_base**(
            torch.arange(0, rotary_dim, 2, dtype=dtype)[:(rotary_dim//2)].float() / rotary_dim
        )
    ) # (rotary_dim//2,)

    positions=torch.arange(context_length, dtype=dtype) # (context_length,)
    angles=positions.unsqueeze(1)*inv_freq.unsqueeze(0) # (context_length, rotary_dim//2)
    angles=torch.cat([angles, angles], dim=-1) # (context_length, rotary_dim)
    # [[c0theta0, c0theta1, c0theta2....c0theta0, c0theta1, c0theta2....],
    #  ...
    #  [cntheta0, cntheta1, cntheta2....cntheta0, cntheta1, cntheta2....],]

    cos=torch.cos(angles)
    sin=torch.sin(angles)
    
    return cos, sin
    

def apply_rope(x, cos, sin, offset=0):
    """
    Args:
        x (torch.Tensor): Input tensor of size (batch_size, num_heads, seq_len, head_dim)
        cos (torch.Tensor): Cosine tensors of size (context_length, rotary_dim)
        sin (torch.Tensor): Sine tensors of size (context_length, rotary_dim)
    """
    _, _, seq_len, head_dim=x.shape
    assert head_dim%2==0, "Head dimension must be even"

    rot_dim=cos.shape[-1] # rotary_dim
    if rot_dim>head_dim: raise ValueError(f"RoPE dim {rot_dim} cannot exceed head_dim {head_dim}")

    x_rot=x[..., :rot_dim]
    x_pass=x[..., rot_dim:]

    x1=x_rot[...,:rot_dim//2]
    x2=x_rot[...,rot_dim//2:]

    cos=cos[offset:(offset+seq_len),:].unsqueeze(0).unsqueeze(0) # (1,1,seq_len,rotary_dim)
    sin=sin[offset:(offset+seq_len),:].unsqueeze(0).unsqueeze(0)

    rotated=torch.cat([-x2,x1], dim=-1)
    x_rotated=(x_rot*cos)+(rotated*sin)

    x_out=torch.cat([x_rotated, x_pass],dim=-1)
    return x_out.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):

        super().__init__()
        assert num_heads % num_kv_groups==0, f"num_heads {num_heads} must be divisible by num_kv_groups {num_kv_groups}"

        self.num_heads=num_heads
        self.num_kv_groups=num_kv_groups
        self.group_size=num_heads//num_kv_groups

        if head_dim is None:
            assert d_in % num_heads==0, f"`d_in` {d_in} must be divisible by `num_heads` {num_heads} if `head_dim` {head_dim} is not set"
            head_dim=d_in//num_heads

        self.head_dim=head_dim
        self.d_out=num_heads*head_dim

        # qwen3.5 full attention uses a gated Q projection (2x output dim)
        self.W_query=nn.Linear(d_in, self.d_out*2, bias=False, dtype=dtype)
        self.W_key=nn.Linear(d_in, num_kv_groups*head_dim, bias=False, dtype=dtype)
        self.W_value=nn.Linear(d_in, num_kv_groups*head_dim, bias=False, dtype=dtype)
        self.out_proj=nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        self.q_norm=self.k_norm=None
        if qk_norm:
            self.q_norm=RMSNorm(head_dim, eps=1e-6)
            self.k_norm=RMSNorm(head_dim, eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens/seq_len, d_in/emb_dim) without cache; with cache, this will be of 
                shape (batch_size, 1, d_in/emb_dim) since num_t
            mask (torch.Tensor): Boolean causal mask of shape (1,1,seq_len, seq_len) without cache and (1,1,1,seq_len+1) with cache
            cos (torch.Tensor): Cosine tensors of size (context_length, rotary_dim)
            sin (torch.Tensor): Sine tensors of size (context_length, rotary_dim)
            cache (tuple[torch.Tensor]): Cache which is a tuple of size two, each of shape (b, num_kv_group, h_seq_len, head_dim),
                where h_seq_len is the length of previously cached tokens
        Returns:
            (torch.Tensor): Output embedding of shape (batch_size, seq_len, d_out=n_heads*head_dim). Note the output shape is similar to
                the output shape from Qwen3_5GateDeltaNet
            (tuple[torch.Tensor]): Next cache, each of shape (b, num_kv_group, h_seq_len_, head_dim), where h_seq_len_ is the updated length
                of previously cached tokens
        """
        b, num_tokens, _=x.shape

        q_and_gate=self.W_query(x) # (batch_size, num_tokens/seq_len, d_out*2) where num_tokens/seq_len=1 for with cache

        # (batch_size, seq_len, num_heads, head_dim*2) where num_tokens/seq_len=1 for with cache
        q_and_gate=q_and_gate.view(b, num_tokens, self.num_heads, self.head_dim*2) 
        queries, gate=torch.chunk(q_and_gate, 2, dim=-1)# each (batch_size, seq_len, num_heads, head_dim) where num_tokens/seq_len=1 for with cache

        gate=gate.reshape(b, num_tokens, self.d_out) # where num_tokens/seq_len=1 for with cache

        keys=self.W_key(x) # (b, seq_len,num_kv_groups*head_dim) without cache; (b, 1,num_kv_groups*head_dim) with cache; 
        values=self.W_value(x) # (b, seq_len,num_kv_groups*head_dim) without cache; (b, 1,num_kv_groups*head_dim) with cache; 
 
        queries=queries.transpose(1,2) # (batch_size, num_heads, seq_len, head_dim) with seq_len=1 for with-cache
        # for keys_new ande values_new, (batch_size, num_kv_group, seq_len, head_dim) with seq_len=1 for with-cache
        keys_new=keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2)  
        values_new=values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2)  

        if self.q_norm: queries=self.q_norm(queries)
        if self.k_norm: keys_new=self.k_norm(keys_new)

        prev_len=0
        if cache is not None:
            prev_k, prev_v=cache # each of size (b, num_kv_group, h_seq_len, head_dim)
            if prev_k is not None:
                prev_len=prev_k.size(2) # seq_len
                keys_cat_raw=torch.cat([prev_k, keys_new], dim=2) # (b, num_kv_group, h_seq_len+1, head_dim)
                values_cat_raw=torch.cat([prev_v, values_new], dim=2) # (b_size, num_kv_group, h_seq_len+1, head_dim)
            else:
                keys_cat_raw=keys_new # (b, num_kv_group, seq_len, head_dim)
                values_cat_raw=values_new # (b, num_kv_group, seq_len, head_dim)
        else:
            keys_cat_raw=keys_new # (b, num_kv_group, seq_len, head_dim)
            values_cat_raw=values_new # (b, num_kv_group, seq_len, head_dim)

        queries=apply_rope(queries, cos, sin, offset=start_pos) # (b, n_heads, seq_len, head_dim) with seq_len=1 for with-cache
        # (b, num_kv_group, seq_len, head_dim) without cache; # (b, num_kv_group, h_seq_len+1, head_dim) with cache
        keys=apply_rope(keys_cat_raw, cos, sin, offset=start_pos-prev_len) 
 
        # for both keys and values below
        # without cache (batch_size, num_kv_group*group_size, seq_len, head_dim) = (batch_size, num_heads, seq_len, head_dim) 
        # with cache (batch_size, num_kv_group*group_size, h_seq_len+1, head_dim) = (batch_size, num_heads, h_seq_len+1, head_dim) 
        keys=keys.repeat_interleave(self.group_size, dim=1)
        values=values_cat_raw.repeat_interleave(self.group_size, dim=1) 
        # for example, before repeat_interleave along dim=1 (query groups)
        # [K1, K2]
        # after repeat_interleave (each query group is repeated group_size times)
        # [K1, K1, K2, K2]
        # if we use regular repeat instead of repeat_interleave, we'd get
        # [K1, K2, K1, K2]

        if cache is not None and cache[0] is not None:
            next_cache=(
                torch.cat([cache[0], keys_new], dim=2), # (batch_size, num_kv_group, h_seq_len+1..., head_dim) 
                torch.cat([cache[1], values_new], dim=2) # (batch_size, num_kv_group, h_seq_len+1..., head_dim) 
            ) # each cache is of size (b, num_kv_group, seq_len, head_dim); keys_new and values_new are (b, num_kv_group, 1, head_dim)
        else: next_cache=(keys_new, values_new) # each (batch_size, num_kv_group, seq_len, head_dim) 

        # for without cache (b, num_heads, seq_len, head_dim)@(batch_size, num_heads, head_dim, seq_len)=(batch_size, num_heads, seq_len, seq_len)
        # for with cache (b, n_heads, 1, head_dim) @(b, n_heads, head_dim, h_seq_len+1) =(b, n_heads, 1, h_seq_len+1) 
        attn_scores=queries@keys.transpose(2,3) 
        # for without cache, mask (1,1,seq_len, seq_len); and for with cache, (1,1,1,h_seq_len+1)
        attn_scores=attn_scores.masked_fill(mask, -torch.inf) 
        attn_weights=torch.softmax(
            attn_scores*(self.head_dim**-0.5), dim=-1, dtype=torch.float32
        ).to(queries.dtype) # without cache (b, n_heads, seq_len, seq_len); with cache (b, n_heads, 1, h_seq_len+1) 

        # for without cache
        # (batch_size, num_heads, seq_len, seq_len)@(batch_size, num_heads, seq_len, head_dim)=(batch_size, num_heads, seq_len, head_dim)
        # (batch_size, num_heads, seq_len, head_dim).transpose(1,2)=(batch_size, seq_len, num_heads, head_dim)
        # for with cache
        # (b, n_heads, 1, h_seq_len+1)@(b, n_heads, h_seq_len+1, head_dim)=(batch_size, num_heads, 1, head_dim)
        # (batch_size, num_heads, 1, head_dim).transpose(1,2)=(batch_size, 1, num_heads, head_dim)
        context=(attn_weights@values).transpose(1,2).reshape(b, num_tokens, self.d_out)

        # qwen3.5 full-attention uses a gatded q projection
        context=context*torch.sigmoid(gate) # (batch_size, seq_len, d_out) where seq_len=1 for with-cache

        return self.out_proj(context), next_cache

