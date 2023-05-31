import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    Taken from : https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
