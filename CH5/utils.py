import os

from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs"""
    def __init__(self, model_path):

        if not os.path.isfile(model_path): raise FileNotFoundError(model_path)

        mergeable=load_tiktoken_bpe(model_path)

        # hard-coded from Meta's tokenizer.json
        self.special={
            "<|begin_of_text|>":128000,
            "<|end_of_text|>":128001,
            "<|start_header_id|>":128006,
            "<|end_header_id|>":128007,
            "<|eot_id|>":128009,
        }
        self.special.update({f"<|reserved_{i}|>":128002+i for i in range(256) if 128002+i not in self.special.values()})
        self.model=tiktoken.Encoding(name=Path(model_path).name, 
                                     pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                                            r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                                            r"|\p{N}{1,3}"
                                            r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                                            r"|\s*[\r\n]+"
                                            r"|\s+(?!\S)"
                                            r"|\s+",
                                     mergeable_ranks=mergeable, 
                                     special_tokens=self.special,
                                    )
    def encode(self, text, bos=False, eos=False, allowed_special=None): # allowed_special is for compatibility
        ids=([self.special["<|begin_of_text|>"]] if bos else []) + self.model.encode(text)
        if eos: ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids): return self.model.decode(ids)



def text_to_token_ids(text, tokenizer, allowed_special={'<|endoftext|>'}):
    """
    Args:
        text (str): Sentence or phrase
    Returns:
        (torch.Tensor): Token indices of size (batch_size, num_tokens) where batch_size is 1
    """
    encoded=tokenizer.encode(text, allowed_special=allowed_special)
    encoded_tensor=torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat=token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def find_highest_gradient(model):
    """A utility function to calculate the highest gradient based on all model weights
    Args:
        model (nn.Module)
    Returns:
        (torch.Tensor): Maximum element value of gradient
    """
    return max(param.grad.flatten().max() for param in model.parameters() if param.grad is not None)
    # max_grad=None
    # for param in model.parameters():
    #     if param.grad is None: continue
    #     grad_values=param.grad.data.flatten()
    #     max_grad_param=grad_values.max()
    #     if max_grad is None or max_grad_param>max_grad: max_grad=max_grad_param
    # return max_grad

def generate(model, idx, max_new_tokens, context_size, temperature=0., top_k=None, eos_id=None):
    """
    Incorporate temperature scaling and top-k and multinomial sampling to generate text
    Args:
        idx (torch.Tensor): Input token indices of size (batch_size, seq_len/num_tokens)
        max_new_tokens (int): The maximum number of new tokens to be generated
        eos_id (torch.Tensor): Index of end-of-sequence token. If provided and detected, stop generating text 
    
    """
    for _ in range(max_new_tokens):

        idx_cond=idx[:, -context_size:] # truncate the token indices to the context-length
        with torch.no_grad(): logits=model(idx_cond) # (batch_size, seq_len/num_tokens, vocab_size)

        logits=logits[:,-1] # only focus on the logit of the last token (batch_size, vocab_size)

        if top_k is not None: # filter logits with top-k sampling
            top_logits,_=torch.topk(logits, top_k) # (batch_size, top_k)
            min_val=top_logits[:,-1:] # (batch_size,1)
            logits=torch.where(logits<min_val, torch.tensor(float('-inf')).to(logits.device), logits) # (batch_size, vocab_size)
        if temperature>0.: # applying temperature scaling
            logits=logits/temperature
            probs=torch.softmax(logits, dim=-1) # (batch_size, vocab_size)
            idx_next=torch.multinomial(probs, num_samples=1) # (batch_size, 1)
        else: idx_next=torch.argmax(logits, dim=-1, keepdim=True) # greedy next token selection

        if idx_next==eos_id: break
        idx=torch.cat((idx, idx_next), dim=-1)

    return idx


def load_weights_into_gpt(gpt, params):

    def assign(left, right):
        assert left.shape==right.shape, f"Shape mismatch. Left:{left.shape}, Right:{right.shape}"
        return torch.nn.Parameter(torch.tensor(right))
    
    # set the model's positional and token embedding weights to those specified in params
    gpt.pos_emb.weight=assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight=assign(gpt.tok_emb.weight, params['wte'])
    
    # iterate over each transformer block
    for block in range(len(params['blocks'])):
    
        # params is of np.ndarray type, so np.split is used to divide the attention and bias weights into 3 equal parts
        # for query, key, and value components
        q_w,k_w,v_w=np.split( # each is of size (emb_dim, emb_dim)=(768,768)
            (params['blocks'][block]['attn']['c_attn'])['w'],3,axis=-1
        )
        gpt.trf_blocks[block].att.W_query.weight=assign(gpt.trf_blocks[block].att.W_query.weight, q_w.T)
        gpt.trf_blocks[block].att.W_key.weight=assign(gpt.trf_blocks[block].att.W_key.weight, k_w.T)
        gpt.trf_blocks[block].att.W_value.weight=assign(gpt.trf_blocks[block].att.W_value.weight, v_w.T)
    
        q_b, k_b, v_b=np.split(
            (params['blocks'][block]['attn']['c_attn'])['b'], 3, axis=-1
        )
        gpt.trf_blocks[block].att.W_query.bias=assign(gpt.trf_blocks[block].att.W_query.bias, q_b)
        gpt.trf_blocks[block].att.W_key.bias=assign(gpt.trf_blocks[block].att.W_key.bias, k_b)
        gpt.trf_blocks[block].att.W_value.bias=assign(gpt.trf_blocks[block].att.W_value.bias, v_b)
        
        gpt.trf_blocks[block].att.out_proj.weight=assign(gpt.trf_blocks[block].att.out_proj.weight, 
                                                         params['blocks'][block]['attn']['c_proj']['w'].T)
        gpt.trf_blocks[block].att.out_proj.bias=assign(gpt.trf_blocks[block].att.out_proj.bias,
                                                       params['blocks'][block]['attn']['c_proj']['b'])
    
        gpt.trf_blocks[block].ff.layers[0].weight=assign(gpt.trf_blocks[block].ff.layers[0].weight,
                                                        params['blocks'][block]['mlp']['c_fc']['w'].T)
        gpt.trf_blocks[block].ff.layers[0].bias=assign(gpt.trf_blocks[block].ff.layers[0].bias,
                                                        params['blocks'][block]['mlp']['c_fc']['b'])
        # layers[1] is GELU
        gpt.trf_blocks[block].ff.layers[2].weight=assign(gpt.trf_blocks[block].ff.layers[2].weight,
                                                        params['blocks'][block]['mlp']['c_proj']['w'].T)
        gpt.trf_blocks[block].ff.layers[2].bias=assign(gpt.trf_blocks[block].ff.layers[2].bias,
                                                        params['blocks'][block]['mlp']['c_proj']['b'])
    
        gpt.trf_blocks[block].norm1.scale=assign(gpt.trf_blocks[block].norm1.scale, params['blocks'][block]['ln_1']['g'])
        gpt.trf_blocks[block].norm1.shift=assign(gpt.trf_blocks[block].norm1.shift, params['blocks'][block]['ln_1']['b'])
        gpt.trf_blocks[block].norm2.scale=assign(gpt.trf_blocks[block].norm2.scale, params['blocks'][block]['ln_2']['g'])
        gpt.trf_blocks[block].norm2.shift=assign(gpt.trf_blocks[block].norm2.shift, params['blocks'][block]['ln_2']['b'])
        
    
    gpt.final_norm.scale=assign(gpt.final_norm.scale, params['g'])
    gpt.final_norm.shift=assign(gpt.final_norm.shift, params['b'])
    gpt.out_head.weight=assign(gpt.out_head.weight, params['wte'])

def assign(left, right, tensor_name='unknown'):
    if left.shape!=right.shape: raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left {left.shape} Right {right.shape}")
    with torch.no_grad():
        if isinstance(right, torch.Tensor): left.copy_(right)
        else: left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
    return left
        
def load_weights_into_llama2(model, param_config, params):
    
    def permute(w:torch.Tensor, n_heads, out_dim, in_dim):
        # The original Meta/Llama checkpoints store Q and K so that the two numbers that form one complex RoPE pair sit next to each other
        # inside the head dimension ("sliced" layout). Our RoPE implementation, similar to the one in Hugging Face, expects an interleaved 
        # layout. For example, with n_heads=2 and head_dim=8
        #                     ┌── pair 0 ──┐      ┌── pair 1 ──┐
        # Meta (sliced): [h0:  r0 r1 r2 r3,    h1:  r0 r1 r2 r3  ]
        # Ours & HF (interleaved): [h0: r0 r0 r1 r1 r2 r2 r3 r3,  h1: ...]    
        # For more information, please see teh discussion in the PR: https://github.com/rasbt/LLMs-from-scratch/pull/747
    
        # So, below, for q_raw and k_raw, we must re-p=order the checkpoint weights using the slices_to_inteleave helper
        # (n_heads, 2, (out_dim//n_heads)//2, in_dim)
        return (w.view(n_heads, (out_dim//n_heads)//2, 2, in_dim)).transpose(1,2).reshape(out_dim, in_dim) 
            
    
    model.tok_emb.weight=assign(model.tok_emb.weight, params['tok_embeddings.weight'])
    for l in range(param_config['n_layers']):
        # The original Meta/Llama checkpoints store Q and K so that the two numbers that form one complex RoPE pair sit next to each other
        # inside the head dimension ("sliced" layout). Our RoPE implementation, similar to the one in Hugging Face, expects an interleaved 
        # layout. For example, with n_heads=2 and head_dim=8
        #                     ┌── pair 0 ──┐      ┌── pair 1 ──┐
        # Meta (sliced): [h0:  r0 r1 r2 r3,    h1:  r0 r1 r2 r3  ]
        # Ours & HF (interleaved): [h0: r0 r0 r1 r1 r2 r2 r3 r3,  h1: ...]    
        # For more information, please see teh discussion in the PR: https://github.com/rasbt/LLMs-from-scratch/pull/747
    
        # So, below, for q_raw and k_raw, we must re-p=order the checkpoint weights using the slices_to_inteleave helper
        q_raw=params[f'layers.{l}.attention.wq.weight']
        model.trf_blocks[l].att.W_query.weight=assign(model.trf_blocks[l].att.W_query.weight, 
                                                      permute(q_raw, param_config['n_heads'], param_config['emb_dim'], param_config['emb_dim']))
        k_raw=params[f"layers.{l}.attention.wk.weight"]
        model.trf_blocks[l].att.W_key.weight=assign(model.trf_blocks[l].att.W_key.weight, 
                                                   permute(k_raw, param_config['n_heads'], param_config['emb_dim'], param_config['emb_dim']))
        model.trf_blocks[l].att.W_value.weight=assign(model.trf_blocks[l].att.W_value.weight, params[f"layers.{l}.attention.wv.weight"])
        model.trf_blocks[l].att.out_proj.weight=assign(model.trf_blocks[l].att.out_proj.weight, params[f"layers.{l}.attention.wo.weight"])
        model.trf_blocks[l].norm1.weight=assign(model.trf_blocks[l].norm1.weight, params[f"layers.{l}.attention_norm.weight"])
        # load feed forward weights
        model.trf_blocks[l].ff.fc1.weight=assign(model.trf_blocks[l].ff.fc1.weight, params[f"layers.{l}.feed_forward.w1.weight"])
        # for some reason w2 and w3 are provided in the wrong order in the weights file
        model.trf_blocks[l].ff.fc2.weight=assign(model.trf_blocks[l].ff.fc2.weight, params[f"layers.{l}.feed_forward.w3.weight"])
        model.trf_blocks[l].ff.fc3.weight=assign(model.trf_blocks[l].ff.fc3.weight, params[f"layers.{l}.feed_forward.w2.weight"])
        model.trf_blocks[l].norm2.weight=assign(model.trf_blocks[l].norm2.weight, params[f"layers.{l}.ffn_norm.weight"])
    
    # load output layer weights
    model.final_norm.weight=assign(model.final_norm.weight, params['norm.weight'])
    model.out_head.weight=assign(model.out_head.weight, params['output.weight'])


def load_weights_into_llama3(model, param_config, params, use_name=False):

    model.tok_emb.weight=assign(model.tok_emb.weight, params['model.embed_tokens.weight'], 'model.embed_tokens.weight' if use_name else None)

    for l in range(param_config['n_layers']):

        # load attention weights
        model.trf_blocks[l].att.W_query.weight=assign(model.trf_blocks[l].att.W_query.weight, 
                                                      params[f"model.layers.{l}.self_attn.q_proj.weight"],
                                                      f"model.layers.{l}.self_attn.q_proj.weight" if use_name else None)
        model.trf_blocks[l].att.W_key.weight=assign(model.trf_blocks[l].att.W_key.weight,
                                                    params[f"model.layers.{l}.self_attn.k_proj.weight"],
                                                    f"model.layers.{l}.self_attn.k_proj.weight" if use_name else None)
        model.trf_blocks[l].att.W_value.weight=assign(model.trf_blocks[l].att.W_value.weight, 
                                                      params[f"model.layers.{l}.self_attn.v_proj.weight"],
                                                      f"model.layers.{l}.self_attn.v_proj.weight" if use_name else None)
        model.trf_blocks[l].att.out_proj.weight=assign(model.trf_blocks[l].att.out_proj.weight, 
                                                      params[f"model.layers.{l}.self_attn.o_proj.weight"],
                                                      f"model.layers.{l}.self_attn.o_proj.weight" if use_name else None)
        model.trf_blocks[l].norm1.weight=assign(model.trf_blocks[l].norm1.weight, params[f'model.layers.{l}.input_layernorm.weight'],
                                                f'model.layers.{l}.input_layernorm.weight' if use_name else None)
        # load feedforward weights
        model.trf_blocks[l].ff.fc1.weight=assign(model.trf_blocks[l].ff.fc1.weight, params[f"model.layers.{l}.mlp.gate_proj.weight"],
                                                 f"model.layers.{l}.mlp.gate_proj.weight" if use_name else None) 
        model.trf_blocks[l].ff.fc2.weight=assign(model.trf_blocks[l].ff.fc2.weight, params[f"model.layers.{l}.mlp.up_proj.weight"],
                                                 f"model.layers.{l}.mlp.up_proj.weight" if use_name else None)
        model.trf_blocks[l].ff.fc3.weight=assign(model.trf_blocks[l].ff.fc3.weight, params[f"model.layers.{l}.mlp.down_proj.weight"],
                                                 f"model.layers.{l}.mlp.down_proj.weight" if use_name else None)
        model.trf_blocks[l].norm2.weight=assign(model.trf_blocks[l].norm2.weight, params[f"model.layers.{l}.post_attention_layernorm.weight"],
                                                f"model.layers.{l}.post_attention_layernorm.weight" if use_name else None)
    # load output layer weights
    model.final_norm.weight=assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight" if use_name else None)

    if "lm_head.weight" in params.keys():
        model.out_head.weight=assign(model.out_head.weight, params['lm_head.weight'], 'lm_head.weight' if use_name else None)
    else: 
        model.out_head.weight=model.tok_emb.weight
        print("Model uses weight tying")

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, label):
    """
    Args:
        epochs_seen (sequence): Training epochs
        tokens_seen (sequence): The number of tokens seen so far
        train_losses (sequence[float]): Training losses
        val_losses (sequence[float]): Validation losses
        label (str): Y-axis label
    """
    fig, ax1=plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.upper())
    ax1.legend(loc="upper right")
    # force the x-axis tick marks to only appears at integer values
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2=ax1.twiny() # create a second x-axis that shared the same y-axis. To create the second y-axis, use twinx()
    ax2.plot(tokens_seen, train_losses, alpha=0) # invisible plot for aligning ticks
    ax2.set_xlabel('Tokens seen')
    fig.tight_layout()