import torch

device = torch.device("cuda", 0)


def to_cuda(tensor: torch.Tensor, use_cuda: bool = True) -> torch.Tensor:
    if use_cuda:
        return tensor.cuda()
    return tensor


def get_tensor(tensor: torch.Tensor, use_cuda=True):
    if use_cuda:
        return tensor.to(device)
    return tensor
