import torch
import torch.nn as nn
import torch.nn.functional as F

# notebook shims for optional fast kernels in transformers
causal_conv1d_fn=None
causal_conv1d_update=None
chunk_gated_delta_rule=None
fused_recurrent_gated_delta_rule=None
FusedRMSNormGated=None
ACT2FN={"silu":F.silu}
is_fast_path_available=False

class _NotebookLogger:
    def __init__(self): self._seen=set()
    def warning_once(self, msg):
        if msg in self._seen: return 
        self._seen.add(msg)
        print(msg)

logger=_NotebookLogger()

# placeholder types for copied annotations
class Qwen3_5Config:
    pass

class Qwen3_5DynamicCache:
    pass

class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon=eps

    def forward(self, hidden_states, gate=None):
        """
        Args:
            hidden_states (torch.Tensor): Hidden features of shape (N, head_dim) where N can be batch_size*seq_len*num_heads
            gate (torch.Tensor): Gate of shape (N, head_dim) where N can be batch_size*seq_len*num_heads
        Returns:
            (torch.Tensor): Scaled hidden states of shape (N, head_dim)
        """
        input_dtype=hidden_states.dtype
        hidden_states=hidden_states.to(torch.float32)
        variance=hidden_states.pow(2).mean(dim=-1, keepdim=True) # (N, head_dim)
        # normalize before gate
        hidden_states=hidden_states*torch.rsqrt(variance+self.variance_epsilon) # (N, head_dim)
        # (head_dim,)(N, head_dim)=(N, head_dim)
        hidden_states=self.weight*hidden_states.to(input_dtype) # allow use of tensor cores (fast gpu calculation if input_dtype is bfloat16)
        # SiLU is prone to numerical underflow/overflow wheb calculated in 16-bit precision, which can cause NaN gradients 
        hidden_states=hidden_states*F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

def apply_mask_to_padding_states(hidden_states, attention_mask, verbose=False):
    """
    Note: this function has never been called since the attention_mask (to linear_attention) is always None
    A safety machanism in attention-free architectures (like mamba or state space models) with padded sequences in a batch. Its job is to zero out
    the padding tokens so that they do not bleed into actual real tokens and corrupt model's generations
    Args:
        hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len) where 1 represents a real token and 0 for a padding token
    CHECK MY DOC
    """
    if verbose: print(f"In apply_mask_to_padding_states: {hidden_states.shape=}, {(attention_mask.shape if attention_mask is not None else None)=}")
    # attention mask is a 2D boolean tensor
    if attention_mask is not None and attention_mask.shape[1]>1 and attention_mask.shape[0]>1:
        # if shape[1] <= 1 (sequence length is 1) or shape[0] <= 1 (batch size is 1), it means there is only a single token or a single sequence 
        # being processed. Single sequences do not require padding, so the function skips the operation to save compute.
        dtype=hidden_states.dtype
        # keep real tokens and force padding tokens to zero
        hidden_states=(hidden_states*attention_mask[:,:,None]).to(dtype)
    return hidden_states


def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """ 1D causal convolution update in sequential architectures like mamba, hyena. A small 1D conv (usually with kernel size/state length of 3/4)
    sliding across the time dimension to mix information from a few adjacent tokens. It uses a rolling cache (conv_state) to store the last few
    tokens' state. It slides the new token in, drops the oldest, and compute conv. 
    
    We note that this function only called when there exists the cache for the linear_attention

    We also note that this function updates `conv_state` so the ` conv_state` after calling this function willl be different than before calling
    this function
    Args:            
        hidden_states (torch.Tensor): Input embedding of size (batch_size, (2*key_dim + value_dim), 1) ~ (batch_size, 3*d_out, 1), often the 
            combination of query, key and value
        conv_state (torch.Tensor): Rolling history buffer of size (batch_size, (2*key_dim + value_dim), conv_kernel_size) ~ 
            (batch_size, 3*d_out, conv_kernel_size), where conv_kernel_size was set to 4 in Qwen3.5. For a conv window of size 4, state_len
            will be 3 (i.e., previous 3 tokens needed to compute a conv with the 1 new token)
        weight (torch.Tensor): Convolution kernel weight of shape ((2*key_dim + value_dim), conv_kernel_size) where the first dimension is 
            number of output channels, and the second dimension is the size of the kernel
        bias (torch.Tensor| None): Convolution kernel bias of shape (2*key_dim + value_dim)
        activation (str|None): Not used here 
    Returns:
        (torch.Tensor): Updated hidden-state with rolling history, having the same shape as input (batch_size, (2*key_dim + value_dim), 1) ~ 
            (batch_size, 3*d_out, 1)
    """
    _, hidden_size, seq_len=hidden_states.shape
    state_len=conv_state.shape[-1] # conv_kernel_size

    # concatenating history with new input
    # cat[(batch_size, (2*key_dim + value_dim), 1), (batch_size, (2*key_dim + value_dim), conv_kernel_size)]=
    #    (batch_size, (2*key_dim + value_dim), conv_kernel_size+1)
    hidden_states_new=torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    # update the rolling cache by removing the old and save the remaining old with a new (batch_size, (2*key_dim + value_dim), conv_kernel_size)
    conv_state.copy_(hidden_states_new[:,:,-state_len:]) 
    # each hidden channel is convolved by its own set of filters
    #    hidden_states_new of size (batch_size, in_channels, conv_kernel_size+1)
    #    weights of size (out_channels, in_channels/hidden_size, conv_kernel_size)
    #    bias of size (out_channels,)
    # depthwise conv operates on each hidden dim separately
    out=F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size) #(batch_size, (2*key_dim + value_dim), out_size)
    # slice the output to match the length of input tokens and apply SiLU (Swish) which is standard in mamba and llama.
    # Note typically seq_len=1
    out=F.silu(out[:,:,-seq_len:]) # (batch_size, (2*key_dim + value_dim), 1) ~ (batch_size, 3*d_out, 1),
    out=out.to(hidden_states.dtype)
    return out

def l2norm(x, dim=-1, eps=1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library"""
    inv_norm=torch.rsqrt((x*x).sum(dim=dim, keepdim=True)+eps)
    return x*inv_norm


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, 
                                 use_qk_l2norm_in_kernel=False):
    """
    This function performs linear attention when there is not rolling state history, used during training and for prefilling. 
    This is the parallel/chunked-based version. Process a sequence using a chunk-parallelized Gated Delta Rule (a linear attention mechanism 
    with dynamic updates and decay), converting sequential operations into hardware-friendly matrix operation
    Args:
        query (torch.Tensor): Query tensor of size (batch, seq_len, num_heads, head_dim)
        key (torch.Tensor): Keuy tensor of size (batch, seq_len, num_heads, head_dim)
        value (torch.Tensor): Value tensor of size (batch, seq_len, num_heads, head_dim)
        g (torch.Tensor): Forget gate/decay tensor, determining how much memory is forgotten or kept over time of size 
            (batch_size, seq_en, num_heads)
        beta (torch.Tensor): Learning rate/update magnitude of the delta rule (delta rule weight), determining the strength of the update to the 
            associative memory (the state) based on the new key-value pair, of size (batch_size, seq_en, num_heads)
        chunk_size (int): Size of local blocks used to parallelize the operation
        initial_state (torch.Tensor): Prior hidden memory state from a previos sequence
        output_final_state (bool): Flag determining whether to return the final recurrent state
    Returns:
        (torch.Tensor): Attention output of size (batch_size, seq_len, num_heads, head_dim)
        (torch.Tensor): Last recurrent state of size (batch_size, num_heads, head_dim, head_dim)
    """
    initial_dtype=query.dtype
    if use_qk_l2norm_in_kernel: 
        query=l2norm(query, dim=-1, eps=1e-6) # (batch, seq_len, num_heads, head_dim)
        key=l2norm(key, dim=-1, eps=1e-6) # (batch, seq_len, num_heads, head_dim)
        
    # swap from (batch, seq_len, num_heads, head_dim) to (batch, num_heads, seq_len, head_dim), 
    # query, key, value to (batch, num_heads, seq_len, head_dim)
    # g and beta to (batch_size, num_heads, seq_en)
    # optimize memory layout and cast to float32 for precision
    query, key, value, beta, g=[x.transpose(1,2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, k_head_dim=key.shape
    v_head_dim=value.shape[-1]

    # padding to match chunk size using 0 to ensure that the sequence length is divisible by chunk_size
    pad_size=(chunk_size-sequence_length%chunk_size) % chunk_size # number of 0 needed to make seq_len divisible by chunk_size
    query=F.pad(query, (0,0,0,pad_size)) # add pad to the bottom rows -- (b,channel,row,cols), i.e., 2D tensor, 
    # giving (batch, num_heads, total_seq_len, head_dim)
    key=F.pad(key, (0,0,0,pad_size)) # add pad to the bottom rows -- (b,channel,row,cols)
    value=F.pad(value, (0,0,0,pad_size)) # add pad to the bottom rows -- (b,channel,row,cols)
    beta=F.pad(beta, (0,pad_size)) #  add pad to the right -- (b,channel,data), i.e., 1D tensor, giving (batch, num_heads, total_seq_len)
    g=F.pad(g, (0,pad_size)) #  add pad to the right -- (b,channel,data)
    total_sequence_length=sequence_length+pad_size

    # apply standard attention scaling factor to the query
    scale=1/(query.shape[-1]**0.5)
    query=query*scale

    # (batch, num_heads, total_seq_len, head_dim) = (batch, num_heads, total_seq_len, head_dim)*(batch, num_heads, total_seq_len, 1)
    v_beta=value*beta.unsqueeze(-1) # multiply value by delta learning rate resulting in a tensor capturing the magnitude of memory update
    k_beta=key*beta.unsqueeze(-1)

    # reshape from (batch, num_heads, total_seq_len, head_dim) to (batch, num_heads, num_chunks, chunk_size, head_dim)
    query, key, value, k_beta, v_beta=[
        x.reshape(x.shape[0], x.shape[1],-1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    # from (batch, num_heads, total_seq_len) to (batch, num_heads, num_chunks, chunk_size)
    g=g.reshape(g.shape[0], g.shape[1], -1, chunk_size) 
        
    # upper triangle mask (including diagonal) to enforce causality within a local chunk, i.e., every elements below diagonal are zero
    mask=torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    # the gate g undergoes a cumulative sum to determine how much information decays between time steps within a chunk
    g=g.cumsum(dim=-1) # (batch, num_heads, num_chunks, chunk_size)
    # pairwise exponential decay factor between all tokens within a chunk, creating a lower-triangle decay matrix exp(g_i - g_j)
    # (g.unsqueeze(-1)-g.unsqueeze(-2)): (b, n_heads, n_chunks, chunk_size,1)-(b, n_heads, n_chunks, 1, chunk_size)
    # gives (b, n_heads, n_chunks, chunk_size,chunk_size)
    decay_mask=((g.unsqueeze(-1)-g.unsqueeze(-2)).tril().exp().float()).tril() # value along diagonal and above are set to 0
    # compute cross-token identity matrix (key*beta @ key^t), then apply decay mask, and mask out future token (upper-triangle) 
    #   k_beta @ key.T (b, n_heads, n_chunks, chunk_size, head_dim) @ (b, n_heads, n_chunks, head_dim, chunk_size)
    #        gives (b, n_heads, n_chunks, chunk_size, chunk_size)
    attn=-((k_beta @ key.transpose(-1,-2)) * decay_mask).masked_fill(mask, 0) # (b, n_heads, n_chunks, chunk_size,chunk_size) with element along
    # diagonal and above are all zero
    for i in range(1, chunk_size): # matrix inversion loop acting as a fast Gauss-Seidel/elimination solver
        row=attn[...,i,:i].clone() # (b, n_heads, n_chunks, :i)
        sub=attn[...,:i,:i].clone() # (b, n_heads, n_chunks, :i, :i)
        # row.unsqueeze(-1)*sub -> (b, n_heads, n_chunks, :i,1)*(b, n_heads, n_chunks, :i,:i)->(b, n_heads, n_chunks, :i,:i)
        # (row.unsqueeze(-1)*sub).sum(-2) -> (b, n_heads, n_chunks, :i)
        attn[...,i,:i]=row+(row.unsqueeze(-1)*sub).sum(-2)
        
    #  (b, n_heads, n_chunks, chunk_size,chunk_size) + (chunk_size,chunk_size)
    attn=attn+torch.eye(chunk_size, dtype=attn.dtype, device=attn.device) # add identity matrix to complete the solver adjustment
    # updated features internal to the chunk after adjusting for local delta update
    # (b, n_heads, n_chunks, chunk_size,chunk_size)@(b, n_heads, n_chunks, chunk_size, head_dim)=(b, n_heads, n_chunks, chunk_size, head_dim)
    value=attn@v_beta 
    # g.exp().unsqueeze(-1) of size (b, n_heads, n_chunks, chunk_size, 1)
    # (k_beta*g.exp().unsqueeze(-1)) = (b, n_heads, n_chunks, chunk_size, head_dim)*(b, n_heads, n_chunks, chunk_size, 1)
    k_cumdecay=attn@(k_beta*g.exp().unsqueeze(-1)) # (b, n_heads, n_chunks, chunk_size, head_dim)
    # inter-chunk recurrent loop
    last_recurrent_state=(
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value) if initial_state is None else initial_state.to(value)
    ) # initialize the memory matrix state (hidden state) to zero if no `initial_state` was provided
    # allocate a blank tensor to store teh final output representations (b, n_heads, n_chunks, chunk_size, head_dim)
    core_attn_out=torch.zeros_like(value) 
    # every along diagonal and below are zero
    mask=torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length//chunk_size):
        # (b, n_heads, chunk_size, head_dim)
        q_i, k_i, v_i=query[:,:,i], key[:,:,i], value[:,:,i] # index along n_chunks dimension to pull out each chunk of query, key, value
        # compute local intra-chunk standard linear attention mapping, scaled by the local chunk decay mask
        #    q_i @ k_i.transpose(-1,-2) (b, n_heads, chunk_size, head_dim)@(b, n_heads, head_dim, chunk_size)=(b, n_heads, chunk_size, chunk_size)
        #    decay_mask (b, n_heads, n_chunks, chunk_size,chunk_size) so decay_mask[:,:,i] (b, n_heads, chunk_size,chunk_size)
        attn=(q_i @ k_i.transpose(-1,-2) * decay_mask[:,:,i]).masked_fill_(mask, 0) # every element above diagonal is set to zero
        # k_cumdecay of size (b, n_heads, n_chunks, chunk_size, head_dim) so k_cumdecay[:,:,i] (b, n_heads, chunk_size, head_dim)
        # and last_recurrent_state of size (b, n_heads, k_head_dim, v_head_dim) where k_head_dim=v_head_dim
        #  (b, n_heads, chunk_size, head_dim)@(b, n_heads, head_dim, head_dim)=(b, n_heads, chunk_size, head_dim)
        v_prime=(k_cumdecay[:,:,i])@last_recurrent_state # old history memory state
        v_new=v_i-v_prime # truely new information (delta error) (b, n_heads, chunk_size, head_dim)
        # chunk output derived from historical memory
        #    g is of size (b, n_heads, n_chunks, chunk_size), so g[:,:,i,:,None] (b, n_heads, chunk_size, 1)
        #    q_i (b, n_heads, chunk_size, head_dim)
        #    last_recurrent_state (b, n_heads, head_dim, head_dim)
        attn_inter=(q_i*g[:,:,i,:,None].exp())@last_recurrent_state # (b, n_heads, chunk_size, head_dim)
        # complete output is the historical reading plus the newly processed local novel information (attn@v_new)
        #   attn@v_new (b, n_heads, chunk_size,chunk_size)@(b, n_heads, chunk_size, head_dim)=(b, n_heads, chunk_size, head_dim)
        #   attn_inter (b, n_heads, chunk_size, head_dim)
        core_attn_out[:,:,i]=attn_inter+attn@v_new #  (b, n_heads, chunk_size, head_dim)
        # update recurrent state for the next chunk to read. It decays the old state completely across the chunk dimension and incorporates
        # the new key-value updates generated during this chunk's window
        #    last_recurrent_state (b, n_heads, head_dim, head_dim)
        #    g is of size (b, n_heads, n_chunks, chunk_size), so g[:,:,i,-1,None,None] gets the last item in chunk_size ->(b, n_heads, 1, 1)
        #    last_recurrent_state*g[:,:,i,-1,None,None].exp() -> (b, n_heads, head_dim, head_dim)
        # 
        #   (g[:,:,i,-1,None]-g[:,:,i]) [(b, n_heads, 1)-(b, n_heads,chunk_size)]->(b, n_heads,chunk_size)
        #   k_i (b, n_heads, chunk_size, head_dim)
        #  (k_i*(g[:,:,i,-1,None]-g[:,:,i]).exp()[...,None]) (b, n_heads, chunk_size, head_dim)
        #  (k_i*(g[:,:,i,-1,None]-g[:,:,i]).exp()[...,None]).transpose(-1,-2)  (b, n_heads, head_dim, chunk_size)
        #  v_new (b, n_heads, chunk_size, head_dim)
        #  out size is (b, n_heads, head_dim, head_dim)
        last_recurrent_state=(
            last_recurrent_state*g[:,:,i,-1,None,None].exp() +
            (k_i*(g[:,:,i,-1,None]-g[:,:,i]).exp()[...,None]).transpose(-1,-2) @ v_new
        ) # (b, n_heads, head_dim, head_dim)

    if not output_final_state: last_recurrent_state=None
    # from (b, n_heads, n_chunks, chunk_size, head_dim) to  (batch, num_heads, (n_chunks*chunk_size)/(seq_len+padding), head_dim)
    core_attn_out=core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1],-1,core_attn_out.shape[-1])
    # (batch, num_heads, seq_len, head_dim)
    core_attn_out=core_attn_out[:,:,:sequence_length] # slice off padding to restore the exact length
    # (batch, seq_len, num_heads, head_dim)
    core_attn_out=core_attn_out.transpose(1,2).contiguous().to(initial_dtype)

    return core_attn_out, last_recurrent_state

def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False):
    """
    This function performing linear attention when there is rolling state history. It is RNN-like version of linear attention, used during inference
    when seq_ln=1
    Args:
        query (torch.Tensor): Tensor of shape (batch_size, seq_len, num_v_heads, head_k_dim) where seq_len=1 
        key (torch.Tensor): Tensor of shape (batch_size, seq_len, num_v_heads, head_k_dim) where seq_len=1
        value (torch.Tensor): Tensor of shape (batch_size, seq_len, num_v_heads, head_v_dim) where seq_len=1
        g (torch.Tensor): Forget gate/decay tensor, determining how much memory is forgotten or kept over time of size,
            of shape (batch_size, seq_len, num_v_heads) where seq_len=1
        beta (torch.Tensor): Learning rate/update magnitude of the delta rule (delta rule weight), determining the strength of the update to the 
            associative memory (the state) based on the new key-value pair, of size (batch_size, seq_en, num_heads) where seq_len=1
        initial_state (torch.Tensor): Previous recurrent state of shape (batch_size, num_heads, k_head_dim, v_head_dim)
        output_final_state (bool): Whether to store the final state
        use_qk_l2norm_in_kernel (bool): Whether to l2 normalize query and key 
    Returns:
        (torch.Tensor): Attention output of size (batch_size, seq_len, num_v_heads, head_v_dim) where seq_len=1
        (torch.Tensor): Last recurrent state of size (batch_size, num_heads, head_k_dim, head_v_dim)
    """ 
    initial_dtype=query.dtype
    if use_qk_l2norm_in_kernel:
        query=l2norm(query, dim=-1, eps=1e-6)
        key=l2norm(key, dim=-1, eps=1e-6)
    # query and key change to (batch_size, num_v_heads, 1,  head_k_dim), similarly value to (batch_size, num_v_heads, 1, head_v_dim)
    # beta and g to (batch_size, num_v_heads, 1)
    query, key, value, beta, g=[
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
        
    batch_size, num_heads, seq_len, k_head_dim=key.shape #only used when seq_len=1
    v_head_dim=value.shape[-1]
    scale=1./(query.shape[-1]**0.5)
    query=query*scale

    core_attn_out=torch.zeros(batch_size, num_heads, seq_len, v_head_dim).to(value)
    last_recurrent_state=(
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value) if initial_state is None else initial_state.to(value)
    )
       
    for i in range(seq_len): # since seq_len=1, this only runs once
        q_t=query[:,:,i] # (batch_size, num_v_heads,  head_k_dim)
        k_t=key[:,:,i] # (batch_size, num_v_heads,  head_k_dim)
        v_t=value[:,:,i] # (batch_size, num_v_heads,  head_v_dim)
        # (batch_size, num_v_heads) -> (batch_size, num_v_heads, 1, 1)
        g_t=g[:,:,i].exp().unsqueeze(-1).unsqueeze(-1) # take the exponential of the gate value
        beta_t=beta[:,:,i].unsqueeze(-1) # (batch_size, num_v_heads, 1)

        # multiply the history memory state by the element-wise decay gate (acting as `forget gate`)
        # (batch_size, num_heads, k_head_dim, v_head_dim)(batch_size, num_v_heads, 1, 1)=(batch_size, num_heads, k_head_dim, v_head_dim)
        last_recurrent_state=last_recurrent_state*g_t 
        # the degree of which history matches the key
        # (batch_size, num_heads, k_head_dim, v_head_dim)(batch_size, num_v_heads,head_k_dim,1)=(batch_size, num_heads, k_head_dim, v_head_dim)
        # (batch_size, num_heads, k_head_dim, v_head_dim).sum(dim=-2)=(batch_size, num_heads, v_head_dim)
        kv_mem=(last_recurrent_state*k_t.unsqueeze(-1)).sum(dim=-2) 
        # error/delta calculation: difference between the target value (v_t) and what the memory predicted (W_{t-1} * k_t). 
        # then scale the error by learning rate/gat beta_t
        # [(batch_size, num_v_heads,  head_v_dim)-batch_size, num_heads, v_head_dim)]*(batch_size, num_v_heads, 1,1)
        # (batch_size, num_v_heads,  head_v_dim)
        delta=(v_t-kv_mem)*beta_t 
            
        # memory update: update the state by forming the outer-product between key (k_t) and error vector (delta). This is the core delta rule 
        # update W_{t-1} + beta_{t}(v_{t} - W_{t-1}k_{t})k_{t}^{T} 
        # (batch_size, num_heads, k_head_dim, v_head_dim)+[(batch_size, num_v_heads,  head_k_dim, 1)*(batch_size, num_v_heads,1,head_v_dim)]
        # = (batch_size, num_heads, k_head_dim, v_head_dim)
        last_recurrent_state=last_recurrent_state+k_t.unsqueeze(-1)*delta.unsqueeze(-2)
        # (batch_size, num_heads, k_head_dim, v_head_dim)*(batch_size, num_v_heads, head_k_dim,1)=(batch_size, num_heads, k_head_dim, v_head_dim)
        # core_attn_out[:,:,i]->(batch_size, num_heads, v_head_dim)
        core_attn_out[:,:,i]=(last_recurrent_state*q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state: last_recurrent_state=None

    # (batch_size, num_heads, seq_len, v_head_dim) to (batch_size, seq_len, num_heads, v_head_dim)
    core_attn_out=core_attn_out.transpose(1,2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
    
# Minimal change: enforce config dtype at the end to avoid bf16/fp32 matmul mismatch in a mixed notebook implementation
class Qwen3_5GateDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size=config.hidden_size
        self.num_v_heads=config.linear_num_value_heads
        self.num_k_heads=config.linear_num_key_heads
        self.head_k_dim=config.linear_key_head_dim
        self.head_v_dim=config.linear_value_head_dim
        self.key_dim=self.head_k_dim*self.num_k_heads # typically d_out, i.e., n_heads*head_dim
        self.value_dim=self.head_v_dim*self.num_v_heads # typically d_out, i.e., n_heads*head_dim

        self.conv_kernel_size=config.linear_conv_kernel_dim
        self.layer_idx=layer_idx
        self.activation=config.hidden_act
        self.act=ACT2FN[config.hidden_act]
        self.layer_norm_epsilon=config.rms_norm_eps

        # QKV
        self.conv_dim=self.key_dim*2 + self.value_dim
        # we note that conv1d is depthwise convolution so its weight is of shape (out_channels, in_channels/groups, kernel)
        # since here in_channel=groups, we have (out_channels, 1, kernel)
        self.conv1d=nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_kernel_size, 
                              groups=self.conv_dim, padding=self.conv_kernel_size-1) # depthwise conv
        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias=nn.Parameter(torch.ones(self.num_v_heads))

        # keep in log space so values are small, but when use exponent it 
        # this makes sure that the scaling exp(A_log) always positive (compared to training and using A directly as a scaling factor) 
        A=torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log=nn.Parameter(torch.log(A))

        self.norm=(
            Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon) if FusedRMSNormGated is None else 
            FusedRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon, activation=self.activation, device=torch.cuda.current_device(),
                              dtype=config.dtype if config.dtype is not None else torch.get_default_dtype())
        )
        self.out_proj=nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.causal_conv1d_fn=causal_conv1d_fn
        self.causal_conv1d_update=causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule=chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )
        self.in_proj_qkv=nn.Linear(self.hidden_size, self.key_dim*2 + self.value_dim, bias=False)
        self.in_proj_z=nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b=nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a=nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Notebook adaptation for dtype consistency
        if config.dtype is not None: self.to(dtype=config.dtype)

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        """
        Args:
            hidden_states (torch.Tensor): Input embedding of shape (batch_size, seq_len, emb_dim/hidden_size), i.e., input to transformer blocks.
                Note that for with-cache calculation, seq_len=1
            cache_params (Qwen3_5LinearAttentionCache)
            cache_position (torch.Tensor): Token position from start to end, e.g., torch.arange(pos_start, pos_end)--in this case of size 
                (pos_end-pos_start). For with-cache calculation, this is of size (1,), e.g., tensor([19]) for the first use of cache for the text
                with sequence length of 19
        Returns:
            (torch.Tensor): Hidden features/embeddings of shape (batch_size, seq_len, emb_dim/hidden_size). Note the output shape is similar to
                the output shape from GroupedQueryAttention
        """
            
        hidden_states=apply_mask_to_padding_states(hidden_states, attention_mask) # (batch_size, seq_len, emb_dim) with seq_len=1 for with cache
 
        # set up dimensions for reshapes later
        batch_size, seq_len, _=hidden_states.shape
        use_precomputed_states=(
            cache_params is not None and cache_params.has_previous_state and seq_len==1 and cache_position is not None
        ) # this is true for when using cache and when cache has the previously computed state of this state

        # getting projected states from cache if it exists
        if cache_params is not None:
            # list of previous conv states (batch_size, (2*key_dim + value_dim), conv_kernel_size) ~ (batch_size, 3*d_out, conv_kernel_size)
            conv_state=cache_params.conv_states[self.layer_idx] 
            # list of previous recurrent states (batch_size, num_heads, k_head_dim, v_head_dim)
            recurrent_state=cache_params.recurrent_states[self.layer_idx] 

        # without cache, (batch_size, seq_len, (2*key_dim + value_dim),) ~ (batch_size, seq_len, 3*d_out,) 
        # with cache, (batch_size, 1, (2*key_dim + value_dim),) ~ (batch_size, 1, 3*d_out,) 
        mixed_qkv=self.in_proj_qkv(hidden_states) 
        # (batch_size, (2*key_dim + value_dim), seq_len) ~ (batch_size, 3*d_out, seq_len) with seq_len=1 for with cache
        mixed_qkv=mixed_qkv.transpose(1,2)

        # (batch_size, seq_len, value_dim,)  ~ (batch_size, d_out, seq_len) with seq_len=1 for with-cache
        z=self.in_proj_z(hidden_states) 
        z=z.reshape(batch_size, seq_len, -1, self.head_v_dim) # (batch_size, seq_len, num_v_heads, head_v_dim) with seq_len=1 for with-cache

        b=self.in_proj_b(hidden_states) # (batch_size, seq_en, num_v_heads) with seq_len=1 for with-cache
        a=self.in_proj_a(hidden_states) # (batch_size, seq_en, num_v_heads) with seq_len=1 for with-cache

        if use_precomputed_states: # this is called when we have rolling history
            # 2. convolution sequence transformation
            # note: the conv state is updated in `causal_conv1d_update` and self.activation is not used in torch_causal_conv1d_update function
            # we also note that conv1d is depthwise convolution so its weight is of shape (out_channels, in_channels/groups, kernel)
            # since here in_channel=groups, we have (out_channels, 1, kernel)
            mixed_qkv=self.causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
        else: # this is called when we do not have rolling history
            if cache_params is not None:
                # cut or pad the input length from the left (history side) so the length = self.conv_kernel_size
                # (batch_size, (2*key_dim + value_dim), conv_kernel_size) ~ (batch_size, 3*d_out, conv_kernel_size)
                conv_state=F.pad(mixed_qkv, (self.conv_kernel_size-mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx]=conv_state

            if self.causal_conv1d_fn is not None:
                mixed_qkv=self.causal_conv1d_fn(x=mixed_qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias, 
                                                activation=self.activation, seq_idx=None)
            else: 
                # mixed_qkv=(batch_size, (2*key_dim + value_dim), seq_len) 
                # conv1d -> (batch_size, (2*key_dim + value_dim), some_length)
                # truncate some_length to seq_len
                mixed_qkv=F.silu(self.conv1d(mixed_qkv)[:,:,:seq_len]) 
                
        # without cache (batch_size, (2*key_dim + value_dim), seq_len)->(batch_size, seq_len, (2*key_dim + value_dim))
        # with cache (batch_size, (2*key_dim + value_dim), 1)->(batch_size, 1, (2*key_dim + value_dim))
        mixed_qkv=mixed_qkv.transpose(1,2)
        # split to (batch_size, seq_len, key_dim), (batch_size, seq_len, key_dim), (batch_size, seq_len, value_dim) where seq_len=1 with with-cache
        query, key, value=torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query=query.reshape(batch_size, seq_len, -1, self.head_k_dim) # (batch_size, seq_len, num_k_heads, head_k_dim) with seq_len=1 for with-cache
        key=key.reshape(batch_size, seq_len, -1, self.head_k_dim) # (batch_size, seq_len, num_k_heads, head_k_dim) with seq_len=1 for with-cache
        value=value.reshape(batch_size, seq_len, -1, self.head_v_dim) # (batch_size, seq_len, num_v_heads, head_v_dim) with seq_len=1 for with-cache

        beta=b.sigmoid() # (batch_size, seq_en, num_v_heads) with seq_len=1 for with-cache
        # if the model is loaded in fp16, without the .float() here, A might be -inf
        # for below, if using cache, seq_len=1
        # dt_bias (num_v_heads,)              _
        #                                      | (a+dt_bias) (batch_size, seq_en, num_v_heads)
        # a (batch_size, seq_len, num_v_heads) _|
        # A_log (num_v_heads,): here e^{A_log} acts as a scaling factor for the decay or gate mechanism
        g=-self.A_log.float().exp() * F.softplus(a.float()+self.dt_bias) # output tensor of size (batch_size, seq_len, num_v_heads)
        if self.num_v_heads//self.num_k_heads>1:
            # if num_k_heads < num_v_heads, expand num_k_heads dim so they match num_v_heads
            query=query.repeat_interleave(self.num_v_heads//self.num_k_heads, dim=2) # (batch_size, seq_len, num_v_heads, head_k_dim)
            key=key.repeat_interleave(self.num_v_heads//self.num_k_heads, dim=2) # (batch_size, seq_len, num_v_heads, head_k_dim)

        if not use_precomputed_states: # call with there is not exist rolling history
            # core_attn_out (batch_size, seq_len, num_v_heads, v_head_dim)
            # last_recurrent_state (batch_size, num_heads, k_head_dim, v_head_dim)
            core_attn_out, last_recurrent_state=self.chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None, 
                                                                            output_final_state=cache_params is not None, 
                                                                            use_qk_l2norm_in_kernel=True)
        else: # call when there is existing rolling history
            # core_attn_out (batch_size, seq_len, num_v_heads, v_head_dim)
            # last_recurrent_state (batch_size, num_heads, k_head_dim, v_head_dim)
            core_attn_out, last_recurrent_state=self.recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                                                                                output_final_state=cache_params is not None, 
                                                                                use_qk_l2norm_in_kernel=True)
            
        # update cache
        if cache_params is not None: cache_params.recurrent_states[self.layer_idx]=last_recurrent_state #(b, n_heads, k_head_dim, v_head_dim)

        # reshape input data into 2D tensor from (b, seq_len, n_v_heads, head_v_dim) to (b*seq_len*n_v_heads, head_v_dim) where seq_len=1 for cache
        core_attn_out=core_attn_out.reshape(-1, self.head_v_dim)
        # from (b_size, seq_len, n_v_heads, head_v_dim) to (b_size*seq_len*n_v_heads, head_v_dim) where seq_len=1 for with-cache
        z=z.reshape(-1, self.head_v_dim) 
        core_attn_out=self.norm(core_attn_out, z) # scaled hidden states of shape (batch_size*seq_len*num_v_heads, head_dim)
        core_attn_out=core_attn_out.reshape(batch_size, seq_len, -1) # (batch_size,seq_len,num_v_heads*head_dim)

        output=self.out_proj(core_attn_out) # from (batch_size,seq_len,num_v_heads*head_dim) to (batch_size, seq_len, emb_dim)

        return output