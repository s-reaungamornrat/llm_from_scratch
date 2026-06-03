import math

import torch

from llm_from_scratch.CH5.optim import generate_and_print_sample
from dpo import compute_dpo_loss_batch, compute_dpo_loss_loader, evaluate_dpo_loss_loader

def train_model(policy_model, reference_model, train_loader, val_loader, optimizer, num_epochs, beta, eval_freq, eval_iter, 
                start_context, tokenizer, warmup_steps, initial_lr=3e-5, min_lr=1e-6, init_beta=None, last_beta=None):
    """
    Args:
        eval_iter (int): Number of evaluation iterations
        eval_freq (int): Frequency of evaluation in iteration units
        warmup_steps (int): The number of warmup steps, typically 0.1% to 20% of the total number of training steps
        init_beta (float): If provided, `beta` will be computed using `init_beta` as a linear increment to `last_beta`
        last_beta (float): If provided, `beta` will be computed using `init_beta` as a linear increment to `last_beta`
    """
    if any(x is not None for x in [init_beta, last_beta]): 
        assert all(x is not None for x in [init_beta, last_beta]) and init_beta<last_beta
        
    # initialize lists to track losses and tokens seen
    tracking={"train_losses":[],
              "train_chosen_rewards":[],
              "train_rejected_rewards":[],
              "val_losses":[],
              "val_chosen_rewards":[],
              "val_rejected_rewards":[],
              "tokens_seen":[],
              "lr":[],
              "beta":[]}
    
    tokens_seen, global_step=0, -1

    # retrieve the maximum learning rate from the optimizer
    peak_lr=optimizer.param_groups[0]['lr']

    # calculate the total number of iterations in the training process
    total_training_steps=len(train_loader)*num_epochs

    # calculate the learning rate increment during warmup phase
    lr_increment=(peak_lr-initial_lr)/warmup_steps

    # calculate beta increment
    beta_increment=(last_beta-init_beta)/total_training_steps if all(x is not None for x in [init_beta, last_beta]) else None

    # main training loop
    for epoch in range(num_epochs):
        policy_model.train()

        for batch in train_loader:

            optimizer.zero_grad() # reset loss gradients from the previous batch iteration
            global_step+=1

            # adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step<warmup_steps: lr=initial_lr+global_step*lr_increment
            else: # cosine annealing after warmup
                progress=(global_step-warmup_steps)/(total_training_steps-warmup_steps)
                lr=min_lr+(peak_lr-min_lr)*0.5*(1.+math.cos(math.pi*progress))
            # apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups: param_group['lr']=lr
            tracking['lr'].append(lr)

            # adjust beta if provided
            if beta_increment is not None:
                beta=init_beta+global_step*beta_increment
            tracking['beta'].append(beta)

            loss, chosen_rewards, rejected_rewards=compute_dpo_loss_batch(batch=batch, policy_model=policy_model, 
                                                                          reference_model=reference_model, beta=beta)
            loss.backward() # calculate loss gradient
            # apply gradient clipping to avoid gradient expose
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=10.)
            
            optimizer.step() # update model weights using loss gradients

            tokens_seen+=batch['chosen'].numel()

            # optional evaluation step
            if global_step%eval_freq==0:
                result=evaluate_dpo_loss_loader(policy_model=policy_model, reference_model=reference_model, train_loader=train_loader,
                                               val_loader=val_loader, beta=beta, eval_iter=eval_iter)
                tracking['train_losses'].append(result['train_loss'])
                tracking['train_chosen_rewards'].append(result['train_chosen_reward'])
                tracking['train_rejected_rewards'].append(result['train_rejected_reward'])
                tracking['val_losses'].append(result['val_loss'])
                tracking['val_chosen_rewards'].append(result['val_chosen_reward'])
                tracking['val_rejected_rewards'].append(result['val_rejected_reward'])
                tracking['tokens_seen'].append(tokens_seen)
                train_reward_margin=result['train_chosen_reward']-result['train_rejected_reward']
                val_reward_margin=result['val_chosen_reward']-result['val_rejected_reward']

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {result['train_loss']:.3f}, Val loss {result['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )
        # print a sample text after each epoch
        generate_and_print_sample(model=policy_model, tokenizer=tokenizer, device=loss.device, start_context=start_context)

    return tracking