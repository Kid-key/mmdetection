
from torch.nn.utils import clip_grad

from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class MyOptimizerHook(Hook):

    def __init__(self, grad_clip=None, mean_grad=False):
        self.grad_clip = grad_clip
        self.mean_grad = mean_grad
        print('myHook')

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def mean_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        for p in params:
            p_shape = p.shape
            if len(p_shape) == 4 and p_shape[2] == p_shape[3] and p_shape[1]>10:
                p_grad = p.grad
                p.grad = p_grad - p_grad.mean([1,2,3], True)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.mean_grad:
            self.mean_grads(runner.model.parameters())
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()

