import math
import torch

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        """
        Args:
            rank (int): A hyperparameter controlling the inner dimension of the matrices A and B, i.e., controlling the number of additional
                parameters. Thus, it is a key factor in determining the balance between model adaptibility and parameter efficiency
            alpha (float): A scaling hyperparameter applied to the output of the low-rank adaptation, controlling the extent to which the adapted
                layer's output is allowed to influence the original output 
        """
        super().__init__()
        self.A=torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) # similar to standard weight initialization
        self.B=torch.nn.Parameter(torch.zeros(rank, out_dim)) # 0 matrix
        self.alpha=alpha
        self.rank=rank

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature of shape (...,in_dim), typically (batch_size, seq_len, in_dim) where in_dim is emb_dim
        Returns:
            (torch.Tensor): Output features with the same shape as the input (batch_size, seq_len, in_dim)
        """
        # Note: the original chapter did not include the scaling by self.rank
        # This scaling is not necessary, but it is more canonical and convenient
        # as this lets us compare runs across different ranks without retuning learning rates
        x=(self.alpha/self.rank)*(x@self.A@self.B) # (batch_size, seq_len, in_dim)
        return x

class LinearWithLoRA(torch.nn.Module):
    """This class can be used to replace existing Linear layers in, for example, self-attention modules or feed-forward modules"""
    def __init__(self, linear, rank, alpha):
        """
        Args:
            rank (int): A hyperparameter controlling the inner dimension of the matrices A and B, i.e., controlling the number of additional
                parameters. Thus, it is a key factor in determining the balance between model adaptibility and parameter efficiency
            alpha (float): A scaling hyperparameter applied to the output of the low-rank adaptation, controlling the extent to which the adapted
                layer's output is allowed to influence the original output 
        """
        super().__init__()
        self.linear=linear
        self.lora=LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    def forward(self, x):
        return self.linear(x)+self.lora(x)

def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # replace the linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)
            
        