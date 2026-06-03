import torch
import torch.nn.functional as F

def compute_dpo_loss(model_chosen_logprobs, model_rejected_logprobs, reference_chosen_logprobs, reference_rejected_logprobs, beta=0.1):
    """Compute the DPO loss to align the model with human preference for a batch of policy and reference model log probabilities
    Args:
        model_chosen_logprobs (torch.Tensor): Log probabilites that the policy model would generate the human-preferred (chosen) text 
            with shape (batch_size,)
        model_rejected_logprobs (torch.Tensor): Log probabilites that the policy model would generate the human-dispreferred (rejected) text
            with shape (batch_size,)
        reference_chosen_logprobs (torch.Tensor): Log probabilites of the reference model for the chosen responses with shape (batch_size,) 
        reference_rejected_logprobs (torch.Tensor): Log probabilites of the reference model for the rejected responses with shape (batch_size,)
        beta (float): Temperature parameter for the DPO loss; In practice, beta is typically set between 0.1 and 0.5. 
            A value of 0.1 is the golden standard starting point for most LLM alignment tasks.
            - High Beta (e.g., 0.5): Acts as a heavy penalty. It keeps the model conservative and close to the reference model.
            - Low Beta (e.g., 0.05 or 0.1): Acts as a loose constraint. It allows the model to change its weights drastically to satisfy the 
                preference dataset, even if it ends up sounding very different from the reference model.
    Returns:
        (tuple[torch.Tensor]): Tuple of 3 scalar tensors (loss, chosen_rewards, rejected_rewards)
    """

    # DPO is trying to widen the gap between `logprobs that model would generate human-preferred text` and `logprobs that model would generate
    # human-dispreferred text`, i.e., it is trying to maximize `model_chosen_logprobs` while minimizing `model_rejected_logprobs`
    # below is from the general relationship of log(a/b)=log(a)-log(b)
    model_logratios=model_chosen_logprobs-model_rejected_logprobs
    reference_logratios=reference_chosen_logprobs-reference_rejected_logprobs
    logits=model_logratios-reference_logratios

    # DPO (Eq.7 of https://arxiv.org/pdf/2305.18290.pdf)
    # we note the sigmoid here estimates the probability that a human would prefer the "chosen" response over the "rejected"
    losses=-F.logsigmoid(beta*logits)

    # optional values to track progress during training
    # we keep track of the chosen_rewards to answer the question "Is the model learning to give higher rewards to the good responses compared to
    # the reference model?
    chosen_rewards=(model_chosen_logprobs-reference_chosen_logprobs).detach()
    # we keep track of the rejected_rewards to answer the question "Is the model learning to give lower rewards to the bad responses compared to 
    # the reference model?
    rejected_rewards=(model_rejected_logprobs-reference_rejected_logprobs).detach()
    # If training is successful, we typically see `chosen_rewards` increases and `rejected_rewards` decreases and becomes negative

    # .mean() to average over the samples in the batch
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def compute_logprobs(logits, labels, selection_mask=None):
    """Compute log probabilities
    Args:
        logits (torch.Tensor): (batch_size, num_tokens, vocab_size)
        labels (torch.Tensor): (batch_size, num_tokens)
        selection_mask (torch.Tensor): Mask to select response section with/without input section without padding (batch_size, num_tokens)
    Returns:
        (torch.Tensor): Mean log probability excluding padding tokens of size (batch_size,)
    """
    # labels are the inputs shifted by one
    labels=labels[:,1:].clone()

    # truncate logits to match the labels num_tokens
    logits=logits[:,:-1]

    log_probs=F.log_softmax(logits, dim=-1)

    # gather the log probabilities for the actual labels
    selected_log_probs=torch.gather(input=log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (batch_size, num_tokens)

    if selection_mask is not None:
        mask=selection_mask[:,1:].clone() # making it the same shape as labels

        # apply the mask to filter out padding tokens
        selected_log_probs=selected_log_probs*mask

        # calculate the average log probability excluding padding tokens
        # this averages over the tokens, so the shape is (batch_size,)
        avg_log_prob=selected_log_probs.sum(-1)/mask.sum(-1)

        return avg_log_prob
        
    return selected_log_probs.mean(-1) # (batch_size, num_tokens)->(batch_size)
        

def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    """Compute the DPO loss on an input batch
    Args:
        batch (dict[str, Any]): Input batch containing
            - 'prompt' (list[torch.Tensor]): List of varying-length instruction tokens (length=number of instructions)
            - 'chosen' (torch.Tensor): A batch of instruction+chosen response of size (batch_size, n_tokens)
            - 'rejected' (torch.Tensor): A batch of instruction+rejected response of size (batch_size, n_tokens)
            - 'rejected_mask' (torch.Tensor): A batch of mask of size (batch_size, n_tokens) where 1 is for padding and
                prompt tokens if `mask_prompt_tokens` is True
            - 'chosen' (torch.Tensor): A batch of mask of size (batch_size, n_tokens) where 1 is for padding and
                prompt tokens if `mask_prompt_tokens` is True
        policy_model (nn.Module): Model that is optimized
        reference_model (nn.Module): Model that is not optimized
        beta (float): Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model
            as beta->0.
    Returns:
        (tuple[torch.Tensor]): Tuple of 3 scalar tensors (loss, chosen_rewards, rejected_rewards)
    """
    policy_chosen_log_probs=compute_logprobs(logits=policy_model(batch['chosen']), # logits (batch,n_tokens,vocab_size)
                                             labels=batch['chosen'], selection_mask=batch['chosen_mask'])
    policy_rejected_log_probs=compute_logprobs(logits=policy_model(batch['rejected']), # logits (batch,n_tokens,vocab_size)
                                               labels=batch['rejected'], selection_mask=batch['rejected_mask'])

    with torch.no_grad():
        ref_chosen_log_probs=compute_logprobs(logits=reference_model(batch['chosen']), # logits (batch,n_tokens,vocab_size)
                                             labels=batch['chosen'], selection_mask=batch['chosen_mask'])
        ref_rejected_log_probs=compute_logprobs(logits=reference_model(batch['rejected']), # logits (batch,n_tokens,vocab_size)
                                             labels=batch['rejected'], selection_mask=batch['rejected_mask'])

    loss, chosen_rewards, rejected_rewards=compute_dpo_loss(model_chosen_logprobs=policy_chosen_log_probs,
                                                            model_rejected_logprobs=policy_rejected_log_probs,
                                                            reference_chosen_logprobs=ref_chosen_log_probs,
                                                            reference_rejected_logprobs=ref_rejected_log_probs,
                                                            beta=beta)
    return loss, chosen_rewards, rejected_rewards

def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):
    """Apply `compute_dpo_loss_batch` to a whole data loader
    Args:
        data_loader (torch.utils.data.Dataloader)
        policy_model (torch.nn.Module): Model being optimized
        reference_model (torch.nn.Module): Model pretrained to have good performance and not being further optimized
        beta (float): Temperature parameter for the DPO loss
        num_batches (int): Number of batches to compute loss on instead of the whole dataset to reduce calculation time
    Returns:
        (tuple[float]): Tuple of average loss, chosen_reward, and rejected_reward as floating-point numbers
    """

    total_loss, total_chosen_rewards, total_rejected_rewards=0., 0., 0.
    if len(data_loader)==0: return float('nan')
    elif num_batches is None: num_batches=len(data_loader)
    else: num_batches=min(num_batches, len(data_loader))

    for i, batch in enumerate(data_loader):
        if i>=num_batches: break
        loss, chosen_rewards, rejected_rewards=compute_dpo_loss_batch(batch=batch, policy_model=policy_model, reference_model=reference_model,
                                                                     beta=beta)
        total_loss+=loss.item()
        total_chosen_rewards+=chosen_rewards.item()
        total_rejected_rewards+=rejected_rewards.item()

    # calculate average
    total_loss/=num_batches
    total_chosen_rewards/=num_batches
    total_rejected_rewards/=num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards

def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """Compute the DPO loss for the training and validation sets
    Args:
        eval_iter (int): Number of batches to evaluate the model on
    Returns:
        (dict[str,float]): Losses and Rewards computed from training and validation sets
    """
    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards=compute_dpo_loss_loader(data_loader=train_loader, policy_model=policy_model,
                                                                                         reference_model=reference_model, beta=beta, 
                                                                                         num_batches=eval_iter)
        val_loss, val_chosen_rewards, val_rejected_rewards=compute_dpo_loss_loader(data_loader=val_loader, policy_model=policy_model,
                                                                                   reference_model=reference_model, beta=beta,
                                                                                   num_batches=eval_iter)
    results={"train_loss":train_loss, 
             "train_chosen_reward": train_chosen_rewards,
             "train_rejected_reward":train_rejected_rewards,
             "val_loss": val_loss,
             "val_chosen_reward": val_chosen_rewards,
             "val_rejected_reward": val_rejected_rewards}
    
    policy_model.train()
    
    return results


