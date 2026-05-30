import torch

def precompute_rope_params(head_dim, theta_base=10000, context_length=4096, freq_config=None):
    """
    In RoPE, positions are encoded using sine and cosine waves of different frequencies (inv_freq). 
        High frequency (short wavelength): encodes local/precise position
        Low frequency (long wavelength): encodes global/distant position
    """

    assert head_dim%2==0, "Embedding dimension must be even"

    # compute theta_i = theta_base ** (-2i / head_dim)
    exponent = torch.arange(0, head_dim, 2)[:(head_dim//2)].float()/head_dim # (head_dim//2,) ranging (0,1), i.e., never = or > 1
    theta_i = theta_base ** exponent # (head_dim//2,) # monotonically increasing function and can be really large
    
    # compute the inverse frequencies in radians per position step (token):  
    # a set of frequency scales applied to different dimensions of the embedding vectors'
    inv_freq=1./ (theta_i) # (head_dim//2,) monotonically descreaing function, from 1 to close to 0. The large theta_base, the faster this decays

    # YARN-based dynamic frequency scaling or RoPE linear/dynamic interpolation. Used to extend the context window of LLM to allow
    # it to process much longer text than it was originally trained on. Instead of stretching all positional frequencies equally, 
    # the code split the frequencies into 3 distinct zones based on their wavelength: low, medium (smooth), high frequencies 
    if freq_config is not None:
        # use two wavelength thresholds/boundaries (low_freq_factor and high_freq_factor) to divide the model's head dimension into three
        # categories
        # zone 1: wavelength>low_freq_wavelen (long distance - spanning across the entire context window): linear scaled
        low_freq_wavelen=freq_config['original_context_length']/freq_config['low_freq_factor'] 
        # zone 2: wavelength < high_freq_wavelen (short distance -- local token relationship) : unscaled
        high_freq_wavelen=freq_config['original_context_length']/freq_config['high_freq_factor']

        # 2pi is radian for 1 cycle * position step (token) per radian ->, so wavelen is in position steps/tokens per cycle
        wavelen=2*torch.pi/inv_freq # cycle * (sec/cycle) -> sec

        inv_freq_llama=torch.where(wavelen>low_freq_wavelen, inv_freq/freq_config['factor'], inv_freq)

        # zone 3: smooth transition zone. Frequencies in the middle connecting "fully scaled" to "completely unscaled". Smooth interpolate
        # between the two states
        is_medium_freq=(wavelen<=low_freq_wavelen) & (wavelen>=high_freq_wavelen)

        # smooth factor is between 0 and 1
        smooth_factor=(freq_config['original_context_length']/wavelen - freq_config['low_freq_factor']) / (
            freq_config['high_freq_factor']-freq_config['low_freq_factor']
        )
        # blend the frequency
        smoothed_inv_freq=(1.-smooth_factor) * (inv_freq/freq_config['factor']) + smooth_factor*inv_freq
        
        inv_freq_llama=torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq=inv_freq_llama
        
    # generate position indices, i.e., an array of integers representing discrete token indices (0,1,2,3...), so its unit is steps (or tokens).
    positions=torch.arange(context_length) # (context_length,)
    
    # compute angles in radians
    angles=positions.unsqueeze(1)*inv_freq.unsqueeze(0) # (context_length,1)*(1,head_dim//2)=(context_length, head_dim//2)
    
    # expand angles to match the head_dim
    angles=torch.cat([angles, angles], dim=-1) # (context_length, head_dim)
    
    # precompute sine and cosine
    cos=torch.cos(angles)
    sin=torch.sin(angles)
    
    return cos, sin

def compute_rope(x, cos, sin):
    """
    Args:
        x (torch.Tensor): Input embedding of shape (batch_size, num_heads, seq_len, head_dim)
        cos (torch.Tensor): cosine of shape (context_length, head_dim)
        sin (torch.Tensor): sine of shape (context_length, head_dim)
    Returns:
        (torch.Tensor): Positional embedded tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    
    batch_size, num_heads, seq_len, head_dim=x.shape
    assert head_dim%2==0, "Head dimension must be even"
    
    # split x into first half and second half
    x1=x[...,:(head_dim//2)] # first half
    x2=x[...,(head_dim//2):] # second half
    
    # adjust sin and cos shapes
    cos=cos[:seq_len].unsqueeze(0).unsqueeze(0) # (1,1,seq_len,head_dim)
    sin=sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
    # apply the rotary transformation
    rotated=torch.cat((-x2,x1), dim=-1)
    x_rotated=(x*cos)+(rotated*sin) # (batch_size, num_heads, seq_len, head_dim)

    return x_rotated.to(dtype=x.dtype)


def calc_model_memory_size(model, input_dtype=torch.float32):
    """Compute the memory size required to store parameters, gradients, and buffers (non-parameters), assuming the element type is the same as 
    input dtype
    Returns:
        (float): Memory required in GB
    """
    
    total_params=total_grads=0
    for param in model.parameters():
        # calculate total number of elements per parameters
        param_size=param.numel()
        total_params+=param_size
        # check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads+=param_size

    # calculate buffer size (non-parameters that require memory)
    total_buffers=sum(buf.numel() for buf in model.buffers())

    # size in bytes=(number of elements) * (size of each element in bytes)
    # we assume parameters, gradients, and buffers are stored in the same type as input dtype
    element_size=torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes=(total_params+total_grads+total_buffers)*element_size

    # convert bytes to gigabytes
    total_memory_gb=total_memory_bytes/(1024**3)

    return total_memory_gb
    