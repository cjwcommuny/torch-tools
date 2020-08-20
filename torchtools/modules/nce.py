import torch
from torch import Tensor

def softmax_nce_loss(X1: Tensor, X2: Tensor, noise_distribution: Tensor) -> Tensor:
    """
    formula: nce_loss = exp(x1 * x2) / (exp(x1 * x2) + sum(exp(x1 * x2_negative)))

    :param X1: shape=(batch_size, vec_len)
    :param X2: shape=(batch_size, vec_len)
    :param noise_distribution: shape=(batch_size, noise_num)
    :return:
        - loss: shape=(batch_size, )
    """
    assert len(X1.shape) == 2
    assert len(X2.shape) == 2
    assert len(noise_distribution.shape) == 2
    assert X1.shape == X2.shape
    assert X1.shape[0] == noise_distribution.shape[0]

    batch_size, vec_len = X1.shape
    noise_num = noise_distribution.shape[1]

    X_negative = X2.index_select(dim=0, index=noise_distribution.view(-1)).view(batch_size, noise_num, vec_len)
    positive_products = torch.einsum("bv,bv->b", X1, X2) # shape=(batch_size, )
    assert positive_products.shape == (batch_size, )
    negative_products = torch.einsum("bnv,bv->bn", X_negative, X1) # shape=(batch_size, noise_num)
    negative_products = negative_products.transpose(0, 1) # shape=(noise_num, batch_size)
    assert negative_products.shape == (noise_num, batch_size)
    positive_negative_products_stack = torch.cat((
        positive_products.unsqueeze(0),
        negative_products
    )) # shape=(1 + noise_num, batch_size)
    assert positive_negative_products_stack.shape == (1 + noise_num, batch_size)
    loss = -positive_products + torch.logsumexp(positive_negative_products_stack, dim=0) # shape=(batch_size, )
    assert loss.shape == (batch_size, )
    return loss