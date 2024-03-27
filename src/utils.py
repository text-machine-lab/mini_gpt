import logging
import torch

def init_weights(model, logger, device):
    """
    Weight initialization is as follows:
    1. Non-layer norm, weight parameters = Xavier
    2. Non-layer norm, bias parameters = constant value of 0.01
    3. Layer norm, weight parameters = constant value of 1
    4. Layer norm, bias parameters = constant value of 0

    NOTE: initializations are performed with fixed seed value

    """
    logger.info(f"Initializing weights of the model...")

    emb_dim = torch.tensor(model.config.hidden_size, device=device)
    num_layers = torch.tensor(model.config.num_hidden_layers, device=device)

    for name_, par_ in model.named_parameters():
        if not (('LayerNorm' in name_) or ('layer_norm') in name_):
            if par_.dim() >= 2:
                # Xavier init for weight parameters
                torch.nn.init.xavier_normal_(par_)
            else:
                # const init for bias
                torch.nn.init.constant_(par_, 0.01)
        else:
            # const init for layer norm
            if 'weight' in name_:
                torch.nn.init.constant_(par_, 1)
            elif 'bias' in name_:
                torch.nn.init.constant_(par_, 0)


    return model