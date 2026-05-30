import torch

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """Accuracy as the percentage of correct predictions"""
    
    model.eval()
    
    correct_predictions, num_examples=0,0
    if num_batches is None: num_batches=len(data_loader)
    else: num_batches=min(num_batches, len(data_loader))
    # input_batch of shape (batch, num_tokens) and target_batch is of shape (batch_size,)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i>=num_batches:break
        input_batch, target_batch=input_batch.to(device), target_batch.to(device)
        with torch.no_grad(): logits=model(input_batch)[:,-1] # logits of the last output token of shape (batch, num_classes)
        predicted_labels=torch.argmax(logits, dim=-1) # of shape (batch_size,)

        num_examples+=predicted_labels.shape[0] # batch dimension
        correct_predictions+=(predicted_labels==target_batch).sum().item()
    return correct_predictions/num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch=input_batch.to(device), target_batch.to(device)
    logits=model(input_batch)[:,-1] # logits of the last output token
    loss=torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss=0.
    if len(data_loader)==0: return float("nan")
    elif num_batches is None: num_batches=len(data_loader)
    else: num_batches=min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i>=num_batches: break
        loss=calc_loss_batch(input_batch, target_batch, model, device)
        total_loss+=loss.item()
    return total_loss/num_batches