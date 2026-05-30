import torch

def calc_loss_batch(input_batch,target_batch,model, device):
    """ Calculate cross entropy loss
    Args:
        input_batch (torch.Tensor): Input token indices of size (batch_size, seq_len/num_tokens) of type int64
        target_batch (torch.Tensor): Target token indices of size (batch_size, seq_len/num_tokens) of type int64
    Returns:
        (torch.Tensor): loss
    """
    input_batch=input_batch.to(device)
    target_batch=target_batch.to(device)
    logits=model(input_batch)
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Compute loss over all batches sampled by the input data loader
    Args:
        data_loader (torch.utils.data.DataLoader)
    Return:
        (float): Average loss over all batches
    """
    total_loss=0.
    if len(data_loader)==0: return float('nan')
    elif num_batches is None: num_batches=len(data_loader) # iterates over all batches if no fixed number of batches is specified
    else: num_batches=min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss=calc_loss_batch(input_batch, target_batch, model, device)
            total_loss+=loss.item()
        else: break
    return total_loss/num_batches