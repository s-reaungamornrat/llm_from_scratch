import os
import json
import requests

import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, mask_instruction=False):
        """
        Args:
            data (dict): Dictionary with format {'instruction':..., 'input':..., 'output':..., }
            mask_instruction (bool): Whether to mask the instruction part in target
        """
        self.data=data
        self.mask_instruction=mask_instruction
        
        # pre-tokenized texts
        self.encoded_texts, self.instruction_token_length=[], []
        for entry in data:
            instruction_plus_input=format_input(entry)
            response_text=f"\n\n### Response:\b{entry['output']}"
            
            instruction_plus_input_tokens=tokenizer.encode(instruction_plus_input)
            response_text_tokens=tokenizer.encode(response_text)

            self.instruction_token_length.append(len(instruction_plus_input_tokens))
            self.encoded_texts.append(instruction_plus_input_tokens+response_text_tokens)
            
            # full_text=instruction_plus_input+response_text
            # self.encoded_texts.append(tokenizer.encode(full_text))
    def __getitem__(self, index): 
        return (self.encoded_texts[index],self.instruction_token_length[index]) if self.mask_instruction else self.encoded_texts[index]
    def __len__(self): return len(self.data)


class PreferenceDataset(Dataset):
    """Instead of a single output sequence/response, the class returns pairs of responses where one is preferred ('chosen') 
    over the other ('rejected)"""
    def __init__(self, data, tokenizer):
        """
        Args:
            data (dict): Dictionary with format {'instruction':..., 'input':..., 'output':..., }
            mask_instruction (bool): Whether to mask the instruction part in target
        """
        self.data=data

        # pre-tokenize texts
        self.encoded_texts=[]
        for entry in data:
            prompt=format_input(entry)
            rejected_response=entry['rejected']
            chosen_response=entry['chosen']

            prompt_tokens=tokenizer.encode(prompt)
            chosen_full_text=f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text=f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_tokens=tokenizer.encode(chosen_full_text)
            rejected_full_tokens=tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({'prompt':prompt_tokens,
                                       'chosen':chosen_full_tokens,
                                       'rejected':rejected_full_tokens})
    
    def __getitem__(self, index): return self.encoded_texts[index]
    def __len__(self): return len(self.data)

        
def preference_collate_fn(batch, pad_token_id=50256, allowed_max_length=None, mask_prompt_tokens=True, device=torch.device('cpu')):
    """
    Args:
        batch (tuple[dict[str,list[int]]]): Sequence of data entry of the format {'prompt':list[int], 'chosen':list[int], 'rejected':list[int]}
            where list[int] is a list of token indices. We note that 'chosen' and 'rejected' contain the tokens of 'prompt' and responses
        pad_token_id (int): Token index of padding
        allowed_max_length (int): Maximum length of tokens allowed. If provide, 'chosen' and 'rejected' token sequences will be truncated if
            longer than this
        mask_prompt_tokens (bool): Whether to mask the prompt (input/instruction into the model)
        device (torch.device): Computing device
    Returns:
        (dict[str, Any]): Output containing the following
            - 'prompt' (list[torch.Tensor]): List of varying-length instruction tokens (length=number of instructions)
            - 'chosen' (torch.Tensor): A batch of instruction+chosen response of size (batch_size, n_tokens)
            - 'rejected' (torch.Tensor): A batch of instruction+rejected response of size (batch_size, n_tokens)
            - 'rejected_mask' (torch.Tensor): A batch of mask of size (batch_size, n_tokens) where 1 is for padding and
                prompt tokens if `mask_prompt_tokens` is True
            - 'chosen' (torch.Tensor): A batch of mask of size (batch_size, n_tokens) where 1 is for padding and
                prompt tokens if `mask_prompt_tokens` is True
                
    Examples: Assuming 8 is the token id of '\n' 
        >>> batch=({'prompt':[8,9,17,6,4],'chosen':[8,9,17,6,4,8,8,1,7,6,6], 'rejected':[8,9,17,6,4,8,8,1,2,3]},
               {'prompt':[3,7,19,6,56,90],'chosen':[3,7,19,6,56,90,8,8,101,76,5,64,43,89], 'rejected':[3,7,19,6,56,90,8,8,1817,975,54]},
               {'prompt':[7,5,54,89,60],'chosen':[7,5,54,89,60,8,8,120,76,54], 'rejected':[7,5,54,89,60,8,8,87,65,546,876,1090]},)
        >>> batch_data=preference_collate_fn(batch, pad_token_id=50256, allowed_max_length=1024, mask_prompt_tokens=True, 
                                         device=torch.device('mps'))
                                 
    """
    # initialize lists to hold batch data
    batch_data={
        "prompt":[],
        "chosen":[],
        "rejected":[],
        "rejected_mask":[],
        "chosen_mask":[]
    }
    
    # determine the longest sequence to set a common padding length
    max_length_common=0
    for key in ['chosen', 'rejected']:
        current_max=max(len(item[key])+1 for item in batch)
        max_length_common=max(max_length_common, current_max)
    
    # process each item in the batch
    for item in batch:
        prompt=torch.tensor(item['prompt'])
        batch_data['prompt'].append(prompt)
    
        for key in ['chosen', 'rejected']:
            # adjust padding according to the common max length
            sequence=item[key]
            padded=sequence+[pad_token_id]*(max_length_common-len(sequence))
            mask=torch.ones(len(padded)).bool()
    
            # set mask for all padding tokens to False
            mask[len(sequence):]=False
    
            # set mask for all input tokens to False
            # set 2 more token-indices to False for the 2 newlines ("\n") tokens before "### Response" 
            if mask_prompt_tokens: mask[:(prompt.shape[0]+2)]=False # i.e., mask is a selection mask for response tokens
    
            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)
    
    # final processing
    for key in ['chosen', 'rejected', 'chosen_mask', 'rejected_mask']:
        # stack all sequences into a tensor for the given key
        tensor_stack=torch.stack(batch_data[key])
    
        # optionally truncate to maximum sequence length
        if allowed_max_length is not None: tensor_stack=tensor_stack[:,:allowed_max_length]
    
        batch_data[key]=tensor_stack.to(device)
    return batch_data  


def _with_instruction_token_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device='cpu'):
    """Form a batch of pairs of inputs and targets from a list of input tokens, without masking out instruction tokens
    Args:
        batch (tuple[list[int]]): Tuple of token-index lists to merge into a batch of an input tensor and a target tensor 
        pad_token_id (int): Token ID for padding, typically corresponding to the token ID of <|endoftext|>
        ignore_index (int): Token ID telling the code to ignore this token in loss calculation
        allowed_max_length (int, optional): If specified, all token ID sequences that are longer than this length will be truncated to this
            length
        device (torch.device): Device to transfer output tensors to
    Returns:
        (torch.Tensor): Input tensor of padded sequence of token IDs
        (torch.Tensor): Target tensor of right-shifted, padded sequence of token IDs
    Examples:
        >>> inputs_1=[0,1,2,3,4]
        >>> inputs_2=[5,6]
        >>> inputs_3=[7,8,9]
        >>> batch=(inputs_1, inputs_2, inputs_3)
        >>> inputs_tensor, targets_tensor=custom_collate_draft_2(batch)
        >>> print(f"{inputs_tensor.shape=}\n\t{inputs_tensor}")
        inputs_tensor.shape=torch.Size([3, 5])
    	tensor([[    0,     1,     2,     3,     4],
            [    5,     6, 50256, 50256, 50256],
            [    7,     8,     9, 50256, 50256]])
        >>> print(f"{targets_tensor.shape=}\n\t{targets_tensor}")
        targets_tensor.shape=torch.Size([3, 5])
        	tensor([[    1,     2,     3,     4, 50256],
                [    6, 50256,  -100,  -100,  -100],
                [    8,     9, 50256,  -100,  -100]])
    """
    # find the longest sequence in the batch and increase the max-length by +1 since targets are created by 
    # shifting a token window to the right by 1 (so we need to pad/append the sequence by a padding)
    batch_max_length=max(len(b)+1 for b in batch)
    
    # pad and prepare inputs
    inputs_list, targets_list=[],[]
    for i, item in enumerate(batch):
        new_item=item.copy()
        # add an <|endoftext|> token
        new_item+=[pad_token_id]
        # pad sequences to batch_max_length
        padded=(
            new_item+[pad_token_id]*(batch_max_length-len(new_item))
        )
        # via padded[:-1], we remove the extra padded token that has been added via the +1 setting in the batch_max_length
        inputs=torch.tensor(padded[:-1]) # (batch_max_length,)
        # shift +1 to the right for targets
        targets=torch.tensor(padded[1:]) # (batch_max_length,)

        # replace all but the first padding tokens in targets by ignore_index
        mask=targets==pad_token_id # (batch_max_length,)
        indices=torch.nonzero(mask).squeeze() # (num_padding,)
        if indices.numel()>1: targets[indices[1:]]=ignore_index # replace all but the first padding with ignore_index

        # optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs=inputs[:allowed_max_length]
            targets=targets[:allowed_max_length]
        
        inputs_list.append(inputs)
        targets_list.append(targets)
    
    # convert list of inputs/targets to tensors and transfer them to target device
    inputs_tensor=torch.stack(inputs_list).to(device)
    targets_tensor=torch.stack(targets_list).to(device)
    
    return inputs_tensor, targets_tensor

def _without_instruction_token_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device='cpu'):
    """Form a batch of pairs of inputs and targets from a list of input tokens, i.e., targets are inputs shifted to the right by one token ID
    Args:
        batch (tuple[tuple[list[int], int]]): Tuple of a pair of token-index lists and the length of instruction tokens to be masked out
        pad_token_id (int): Token ID for padding, typically corresponding to the token ID of <|endoftext|>
        ignore_index (int): Token ID telling the code to ignore this token in loss calculation
        allowed_max_length (int, optional): If specified, all token ID sequences that are longer than this length will be truncated to this
            length
        device (torch.device): Device to transfer output tensors to
    Returns:
        (torch.Tensor): Input tensor of padded sequence of token IDs
        (torch.Tensor): Target tensor of right-shifted, padded sequence of token IDs
    Examples:
        >>> inputs_1=[0,1,2,3,4]
        >>> inputs_2=[5,6]
        >>> inputs_3=[7,8,9]
        >>> batch=(inputs_1, inputs_2, inputs_3)
        >>> inputs_tensor, targets_tensor=custom_collate_draft_2(batch)
        >>> print(f"{inputs_tensor.shape=}\n\t{inputs_tensor}")
        inputs_tensor.shape=torch.Size([3, 5])
    	tensor([[    0,     1,     2,     3,     4],
            [    5,     6, 50256, 50256, 50256],
            [    7,     8,     9, 50256, 50256]])
        >>> print(f"{targets_tensor.shape=}\n\t{targets_tensor}")
        targets_tensor.shape=torch.Size([3, 5])
        	tensor([[    1,     2,     3,     4, 50256],
                [    6, 50256,  -100,  -100,  -100],
                [    8,     9, 50256,  -100,  -100]])
    """
    # find the longest sequence in the batch and increase the max-length by +1 since targets are created by 
    # shifting a token window to the right by 1 (so we need to pad/append the sequence by a padding)
    batch_max_length=max(len(b[0])+1 for b in batch)
    
    # pad and prepare inputs
    inputs_list, targets_list=[],[]
    for (item, ln) in batch: # each element in the batch is a pair of tokens and the instuction length
        new_item=item.copy()
        # add an <|endoftext|> token
        new_item+=[pad_token_id]
        # pad sequences to batch_max_length
        padded=(
            new_item+[pad_token_id]*(batch_max_length-len(new_item))
        )
        # via padded[:-1], we remove the extra padded token that has been added via the +1 setting in the batch_max_length
        inputs=torch.tensor(padded[:-1]) # (batch_max_length,)
        # shift +1 to the right for targets
        targets=torch.tensor(padded[1:]) # (batch_max_length,)

        # replace all but the first padding tokens in targets by ignore_index
        mask=targets==pad_token_id # (batch_max_length,)
        indices=torch.nonzero(mask).squeeze() # (num_padding,)
        if indices.numel()>1: targets[indices[1:]]=ignore_index # replace all but the first padding with ignore_index

        # replace all instruction tokens in the targets by ignore_index. We note that the targets are from shifting the 
        # inputs (instructions) to the right by 1 token so we delete the instruction length by 1
        targets[:(ln-1)]=ignore_index
        
        # optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs=inputs[:allowed_max_length]
            targets=targets[:allowed_max_length]
        
        inputs_list.append(inputs)
        targets_list.append(targets)
    
    # convert list of inputs/targets to tensors and transfer them to target device
    inputs_tensor=torch.stack(inputs_list).to(device)
    targets_tensor=torch.stack(targets_list).to(device)
    
    return inputs_tensor, targets_tensor

    
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device=torch.device('cpu')):
    """Form a batch of pairs of inputs and targets from a list of input tokens, i.e., targets are inputs shifted to the right by one token ID
    Args:
        batch (tuple[list[int]] | tuple[tuple[list[int], int]]): Tuple of token-index lists to merge into a batch of 
            an input tensor and a target tensor or tuple of a pair of token-index lists and the length of instruction tokens to be masked out
        pad_token_id (int): Token ID for padding, typically corresponding to the token ID of <|endoftext|>
        ignore_index (int): Token ID telling the code to ignore this token in loss calculation
        allowed_max_length (int, optional): If specified, all token ID sequences that are longer than this length will be truncated to this
            length
        device (torch.device): Device to transfer output tensors to. This allows device transfer to be done as a background process 
            (outside a training loop), thus preventing it from blocking the GPU during model training
    Returns:
        (torch.Tensor): Input tensor of padded sequence of token IDs
        (torch.Tensor): Target tensor of right-shifted, padded sequence of token IDs
    Examples:
        >>> inputs_1=[0,1,2,3,4]
        >>> inputs_2=[5,6]
        >>> inputs_3=[7,8,9]
        >>> batch=(inputs_1, inputs_2, inputs_3)
        >>> inputs_tensor, targets_tensor=custom_collate_draft_2(batch)
        >>> print(f"{inputs_tensor.shape=}\n\t{inputs_tensor}")
        inputs_tensor.shape=torch.Size([3, 5])
    	tensor([[    0,     1,     2,     3,     4],
            [    5,     6, 50256, 50256, 50256],
            [    7,     8,     9, 50256, 50256]])
        >>> print(f"{targets_tensor.shape=}\n\t{targets_tensor}")
        targets_tensor.shape=torch.Size([3, 5])
        	tensor([[    1,     2,     3,     4, 50256],
                [    6, 50256,  -100,  -100,  -100],
                [    8,     9, 50256,  -100,  -100]])
    """
    if len(batch[0])==2 and isinstance(batch[0][-1], int): # mask out instruction tokens
        return _without_instruction_token_collate_fn(batch, pad_token_id=pad_token_id, ignore_index=ignore_index,
                                                     allowed_max_length=allowed_max_length, device=device)
    else: return _with_instruction_token_collate_fn(batch, pad_token_id=pad_token_id, ignore_index=ignore_index, 
                                                    allowed_max_length=allowed_max_length, device=device)


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        response=requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        text_data=response.text
        with open(file_path, 'w', encoding='utf-8') as file: file.write(text_data)

    with open(file_path, 'r', encoding='utf-8') as file: data=json.load(file)
    return data
    
def format_input(entry):
    instruction_text=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text=f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    return instruction_text+input_text

def decode_tokens_from_batch(token_ids, tokenizer):
    ids_in_python_list=token_ids.flatten().tolist()
    return tokenizer.decode(ids_in_python_list)


def extract_response(response_text, input_text):
    """
    Args:
        response_text (str): Output from model with instruction and response 
        input_text (str): Input to the model comprising instruction and/or input
    Returns:
        (str): Response after removing prompt formatting
    """
    return response_text[len(input_text):].replace("### Response:", "").strip()