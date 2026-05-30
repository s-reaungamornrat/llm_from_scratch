import torch
from llm_from_scratch.CH5.qwen35.qwen import KVCache

def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    """
    We do stream to reduce percieved latency (users can start reading right away) and also for memory management since we do not have to store
    a growing list of tokens in memory. 
    """

    model.eval()
    with torch.no_grad():
        # initialize a storage object for key and value vectors
        cache=KVCache(n_layers=model.cfg['n_layers']) #storing the hidden states of previous tokens so the model doesn't have to re-calculate them 
        
        model.reset_kv_cache() # ensure that the internal cache state is clean before starting a new generation

        # prime the cache with the initial context by feeding the entire prompt (token_ids) to the model. 
        # The model fills the cache with prompt's data and return logits (prediction scores) for the next token
        logits=model(token_ids, cache=cache) # (batch_size, seq_len, vocab_size)

        # generation loop
        for i in range(max_new_tokens): # A loop to limit how long the model will talk so it does not talk forever
            # logits[:,-1]=(batch_size, vocab_size) i.e., we only care about the predictions for the very last token in the sequence.
            next_token=torch.argmax(logits[:,-1], dim=-1, keepdim=True)# greedily pick a single most likely token index

            if eos_token_id is not None and torch.all(next_token==eos_token_id): # if model picks `end of sequence`, we stop immediately
                break
            yield next_token # pause and send the new token out to the caller immediately

            token_ids=torch.cat([token_ids, next_token], dim=1) # append the new token to history
            # feed only the next token (1 single index) to the model, because cache already holds history/math of the previous tokens
            logits=model(next_token, cache=cache)

def load_weights_into_qwen3_5(model, param_config, params):
    """
    Args:
        model (torch.nn.Module):
        param_config (dict[str, Any]): Model configuration/settings 
        params (dict[str, torch.Tensor]): Model parameter values
    """

    def assign(left, right, tensor_name='unknown'):
        """Assign values of the right tensor to the left tensor"""
        assert left.shape==right.shape, f"Shape mismatch for tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"

        with torch.no_grad():
            if isinstance(right, torch.Tensor): left.copy_(right)
            else: left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
        return left       


    if "model.embed_tokens.weight" in params: model_prefix="model"
    elif "model.language_model.embed_tokens.weight" in params: model_prefix="model.language_model"
    else: raise KeyError("Could not find embed token weights in checkpoint")

    def pkey(suffix): 
        """Forming a proper parameter name by prepending prefix to the input suffix"""
        return f"{model_prefix}.{suffix}"

    model.tok_emb.weight=assign(model.tok_emb.weight, params[pkey("embed_tokens.weight")], pkey("embed_tokens.weight"))
    n_layers=param_config["n_layers"]
    layer_types=param_config.get("layer_types", ["full_attention"]*n_layers)

    for l in range(n_layers):
        block=model.trf_blocks[l]
        layer_type=layer_types[l]

        if layer_type=='full_attention':
            att=block.token_mixer
            att.W_query.weight=assign(att.W_query.weight, params[pkey(f"layers.{l}.self_attn.q_proj.weight")], 
                                      pkey(f"layers.{l}.self_attn.q_proj.weight"))
            att.W_key.weight=assign(att.W_key.weight, params[pkey(f"layers.{l}.self_attn.k_proj.weight")],
                                    pkey(f"layers.{l}.self_attn.k_proj.weight"))
            att.W_value.weight=assign(att.W_value.weight, params[pkey(f"layers.{l}.self_attn.v_proj.weight")],
                                      pkey(f"layers.{l}.self_attn.v_proj.weight"))
            att.out_proj.weight=assign(att.out_proj.weight, params[pkey(f"layers.{l}.self_attn.o_proj.weight")],
                                       pkey(f"layers.{l}.self_attn.o_proj.weight"))
            if hasattr(att, "q_norm") and att.q_norm is not None:
                att.q_norm.weight=assign(att.q_norm.weight, params[pkey(f"layers.{l}.self_attn.q_norm.weight")],
                                         pkey(f"layers.{l}.self_attn.q_norm.weight"))
            if hasattr(att, "k_norm") and att.k_norm is not None:
                att.k_norm.weight=assign(att.k_norm.weight, params[pkey(f"layers.{l}.self_attn.k_norm.weight")],
                                         pkey(f"layers.{l}.self_attn.k_norm.weight"))
        elif layer_type=="linear_attention":
            lat=block.token_mixer
            lat.dt_bias=assign(lat.dt_bias, params[pkey(f"layers.{l}.linear_attn.dt_bias")], pkey(f"layers.{l}.linear_attn.dt_bias"))
            lat.A_log=assign(lat.A_log, params[pkey(f"layers.{l}.linear_attn.A_log")], pkey(f"layers.{l}.linear_attn.A_log"))
            lat.conv1d.weight=assign(lat.conv1d.weight, params[pkey(f"layers.{l}.linear_attn.conv1d.weight")], 
                                     pkey(f"layers.{l}.linear_attn.conv1d.weight"))
            lat.norm.weight=assign(lat.norm.weight, params[pkey(f"layers.{l}.linear_attn.norm.weight")],pkey(f"layers.{l}.linear_attn.norm.weight"))
            lat.out_proj.weight=assign(lat.out_proj.weight, params[pkey(f"layers.{l}.linear_attn.out_proj.weight")],
                                       pkey(f"layers.{l}.linear_attn.out_proj.weight"))
            lat.in_proj_qkv.weight=assign(lat.in_proj_qkv.weight, params[pkey(f"layers.{l}.linear_attn.in_proj_qkv.weight")],
                                          pkey(f"layers.{l}.linear_attn.in_proj_qkv.weight"))
            lat.in_proj_z.weight=assign(lat.in_proj_z.weight, params[pkey(f"layers.{l}.linear_attn.in_proj_z.weight")],
                                        pkey(f"layers.{l}.linear_attn.in_proj_z.weight"))
            lat.in_proj_b.weight=assign(lat.in_proj_b.weight, params[pkey(f"layers.{l}.linear_attn.in_proj_b.weight")],
                                        pkey(f"layers.{l}.linear_attn.in_proj_b.weight"))
            lat.in_proj_a.weight=assign(lat.in_proj_a.weight, params[pkey(f"layers.{l}.linear_attn.in_proj_a.weight")],
                                        pkey(f"layers.{l}.linear_attn.in_proj_a.weight"))
        else: raise ValueError(f"Unsupported layer type: {layer_type}")
            
        block.norm1.weight=assign(block.norm1.weight, params[pkey(f"layers.{l}.input_layernorm.weight")],pkey(f"layers.{l}.input_layernorm.weight"))
        block.ff.fc1.weight=assign(block.ff.fc1.weight, params[pkey(f"layers.{l}.mlp.gate_proj.weight")], pkey(f"layers.{l}.mlp.gate_proj.weight"))
        block.ff.fc2.weight=assign(block.ff.fc2.weight, params[pkey(f"layers.{l}.mlp.up_proj.weight")],pkey(f"layers.{l}.mlp.up_proj.weight"))
        block.ff.fc3.weight=assign(block.ff.fc3.weight, params[pkey(f"layers.{l}.mlp.down_proj.weight")], pkey(f"layers.{l}.mlp.down_proj.weight"))
        block.norm2.weight=assign(block.norm2.weight, params[pkey(f"layers.{l}.post_attention_layernorm.weight")],
                                  pkey(f"layers.{l}.post_attention_layernorm.weight"))
    model.final_norm.weight=assign(model.final_norm.weight, params[pkey("norm.weight")], pkey("norm.weight"))

    if "lm_head.weight" in params:
        model.out_head.weight=assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    elif pkey("lm_head.weight") in params:
        model.out_head.weight=assign(model.out_head.weight, params[pkey("lm_head.weight")], pkey("lm_head.weight"))
    else:
        model.out_head.weight=model.tok_emb.weight
        print("Model uses weight tying")