import torch

# https://arxiv.org/pdf/1812.06127.pdf
def proximal_term(mu, local_model, global_model):
    """Proximal Term of FedProx
    """     
    local_parameters = list(local_model.parameters())
    global_parameters = list(global_model.parameters())
    
    # Square of the l2_norm
    sq_l2_norm = sum([torch.sum((local_parameters[i]-global_parameters[i])**2) 
        for i in range(len(local_parameters))])
    
    return (mu/2) * sq_l2_norm