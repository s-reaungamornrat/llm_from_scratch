import math
import torch
import torch.nn as nn

from llm_from_scratch.CH5.loss import calc_loss_batch, calc_loss_loader
from llm_from_scratch.CH4.gpt import generate_text_simple
from llm_from_scratch.CH5.utils import text_to_token_ids, token_ids_to_text

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, 
                       start_context, tokenizer):
    """
    Args:
        eval_iter (int): Number of evaluation iterations
        eval_freq (int): Frequency of evaluation in iteration unit
    """
    # initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen=[],[],[]
    tokens_seen, global_step=0,-1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen+=input_batch.numel()
            global_step+=1

            if global_step%eval_freq==0:
                train_loss, val_loss=evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        # print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter=None):
    """
    Args:
        eval_iter (int): Number of evaluation iterations. If None, use the whole batch
    """
    # disable dropout during evaluation for stable and reproducible results
    model.eval()
    with torch.no_grad(): # disable gradient tracking 
        train_loss=calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss=calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size=model.pos_emb.weight.shape[0]
    encoded=text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids=generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text=token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) # compact print format
    model.train()


def train_model(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, start_context,
               tokenizer, warmup_steps, initial_lr=3e-5, min_lr=1e-6):
    """
    Args:
        eval_iter (int): Number of evaluation iterations
        eval_freq (int): Frequency of evaluation in iteration unit
    """
    train_losses, val_losses, track_tokens_seen, track_lrs=[],[],[],[]
    tokens_seen, global_step=0,-1

    # retrieve the maximum learning rate from the optimizer
    peak_lr=optimizer.param_groups[0]['lr']

    # calculate the total number of iterations in the training process
    total_training_steps=len(train_loader)*n_epochs

    # calculate the learning rate increment during warmup phase
    lr_increment=(peak_lr-initial_lr)/warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step+=1

            # adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step<warmup_steps: lr=initial_lr+global_step*lr_increment
            else: # cosine annealing after warmup
                progress=(global_step-warmup_steps)/(total_training_steps-warmup_steps)
                lr=min_lr+(peak_lr-min_lr)*0.5*(1.+math.cos(math.pi*progress))
            # apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups: param_group['lr']=lr
            track_lrs.append(lr) # store current learning rate

            # calculate and backpropagate loss
            loss=calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # apply gradient clipping after the warmup phase to avoid gradient explose
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)

            optimizer.step()
            tokens_seen+=input_batch.numel()

            # periodically evaluate the model on the training and validation sets
            if global_step%eval_freq==0:
                train_loss, val_loss=evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                # print the current losses
                print(f"EP {epoch+1} (Iter {global_step:06d}): Train loss {train_loss:.3f} Val loss {val_loss:.3f}")

        # generate and print a sample from the model to monitor prgress
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen, track_lrs