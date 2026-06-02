import math
import torch
import matplotlib.pyplot as plt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label='loss'):
    """
    Args:
        epochs_seen (sequence): List of epochs with the same size as `train_values` and `val_values`
        examples_seen (sequence): List of numbers of seen examples with the same size as `train_values` and `val_values`
        train_values (sequence): List of train values, either losses or accuracies
        val_values (sequence): List of validation values, either losses or accuracies
        label (str): Label of values, either loss or accuracy
    """
    fig, ax1=plt.subplots(figsize=(5,3))

    # plot training and validation values against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # create a second x-axis for examples seen
    ax2=ax1.twiny() # create the 2nd x-axis sharing the same y
    ax2.plot(examples_seen, train_values, alpha=0) # invisible plot for aligning ticks

    fig.tight_layout()

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

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss=calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss=calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
    
def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter,
                             warmup_steps, initial_lr=3e-5, min_lr=1e-6):
    """
    Args:
        eval_iter (int): Number of evaluation iterations
        eval_freq (int): Frequency of evaluation in iteration unit
        warmup_steps (int): Typically the number of warm-up steps is 0.1% or 20% of the total number of steps
    """
    # initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs=[],[],[],[]
    examples_seen, global_step=0,-1
    track_lrs=[]

    # retrieve the maximum learning rate from the optimizer
    peak_lr=optimizer.param_groups[0]['lr']

    # calculate the total number of iterations in the training process
    total_training_steps=len(train_loader)*num_epochs

    # calculate the learning rate increment during warmup phase
    lr_increment=(peak_lr-initial_lr)/warmup_steps

    # main training loop
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # reset loss gradients from previous batch iteration
            global_step+=1

            # adjust the learning rate based on the current phae (warmup or cosine annealing)
            if global_step<warmup_steps: lr=initial_lr+global_step*lr_increment
            else: # cosine annealing after warmup
                progress=(global_step-warmup_steps)/(total_training_steps-warmup_steps)
                lr=min_lr+(peak_lr-min_lr)*0.5*(1.+math.cos(math.pi*progress))
            # apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups: param_group['lr']=lr
            track_lrs.append(lr)
                
            loss=calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calculate loss gradients
            optimizer.step()

            # apply gradient clipping to avoid gradient explose
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            
            examples_seen+=input_batch.shape[0] # track examples instead of tokens
            

            # optional evaludation step
            if global_step%eval_freq==0:
                train_loss, val_loss=evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        # calculate accuracy after each epoch
        train_accuracy=calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy=calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Train accuracy: {train_accuracy*100:.2f}% | Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
    return train_losses, val_losses, train_accs, val_accs, examples_seen, track_lrs