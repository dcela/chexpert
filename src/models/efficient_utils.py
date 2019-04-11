# This implementation is a new efficient implementation of Densenet-BC,
# as described in "Memory-Efficient Implementation of DenseNets",
# an upgrade version of https://github.com/gpleiss/efficient_densenet_pytorch/tree/pytorch_0.3.1
from functools import reduce
from operator import mul
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _single, _pair, _triple


class EfficientDensenetBottleneck(nn.Module):
    """
    A optimized layer which encapsulates the batch normalization, ReLU, and
    convolution operations within the bottleneck of a DenseNet layer.

    This layer usage shared memory allocations to store the outputs of the
    concatenation and batch normalization features. Because the shared memory
    is not perminant, these features are recomputed during the backward pass.
    """

    def __init__(self, num_input_channels, num_output_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, dims=2,
                 momentum=0.1, eps=1e-5):
        super(EfficientDensenetBottleneck, self).__init__()
        assert dims in [1, 2, 3]
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.dims = dims
        _repeat = dims == 1 and _single or dims == 2 and _pair or dims == 3 and _triple
        self.kernel_size = _repeat(kernel_size)
        self.stride = _repeat(stride)
        self.padding = _repeat(padding)
        self.dilation = _repeat(dilation)
        self.groups = groups
        self.bias = bias
        self.momentum = momentum
        self.eps = eps
        self.conv = self.dims == 1 and F.conv1d or self.dims == 2 and F.conv2d or self.dims == 3 and F.conv3d
        self.register_parameter('norm_weight', nn.Parameter(torch.Tensor(num_input_channels)))
        self.register_parameter('norm_bias', nn.Parameter(torch.Tensor(num_input_channels)))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.register_buffer('norm_num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_parameter('conv_weight', nn.Parameter(torch.Tensor(num_output_channels,
                                                                         num_input_channels, *self.kernel_size)))
        if bias:
            self.register_parameter('conv_bias', nn.Parameter(torch.Tensor(num_output_channels)))
        else:
            self.register_parameter('conv_bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        self.norm_running_mean.zero_()
        self.norm_running_var.fill_(1)
        self.norm_num_batches_tracked.zero_()
        self.norm_weight.data.uniform_()
        self.norm_bias.data.zero_()
        n = self.num_input_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(self.num_input_channels)
        self.conv_weight.data.uniform_(-stdv, stdv)
        if self.conv_bias is not None:
            self.conv_bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, shared_alloc):
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        if inputs[0].dim() != self.dims + 2:
            raise ValueError('expected {}D input (got {}D input)'
                             .format(self.dims + 2, inputs[0].dim()))
        exponential_average_factor = 0.0
        if self.training:
            self.norm_num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.norm_num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        # The EfficientDensenetBottleneckFn performs the concatenation, batch norm, and ReLU.
        # It does not create any new storage
        # Rather, it uses a shared memory allocation to store the intermediate feature maps
        # These intermediate feature maps have to be re-populated before the backward pass
        fn = _EfficientDensenetBottleneckFn(shared_alloc,
                                            self._buffers['norm_running_mean'], self._buffers['norm_running_var'],
                                            training=self.training, momentum=exponential_average_factor, eps=self.eps)
        relu_output = fn(self._parameters['norm_weight'], self._parameters['norm_bias'], *inputs)

        # The convolutional output - using relu_output which is stored in shared memory allocation
        conv_output = self.conv(relu_output, self._parameters['conv_weight'], bias=self._parameters['conv_bias'],
                                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        # Register a hook to re-populate the storages (relu_output and concat) on backward pass
        # To do this, we need a dummy function
        output = _DummyBackwardHook.apply(conv_output, fn)

        # Return the convolution output
        return output

    def __repr__(self):
        s = ('{name}(Concat'
             '\n\t--> BatchNorm({num_input_channels}, momentum={momentum}, eps={eps}) --> ReLU(inplace=True)'
             '\n\t--> Conv{dims}d({num_input_channels}, {num_output_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.bias:
            s += ', bias=False'
        s += '))'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class _EfficientDensenetBottleneckFn(Function):
    """
    The autograd function which performs the efficient bottlenck operations:
    --
    1) concatenation
    2) Batch Normalization
    3) ReLU
    --
    Convolution is taken care of in a separate function

    NOTE:
    The output of the function (ReLU) is written on a temporary memory allocation.
    If the output is not used IMMEDIATELY after calling forward, it is not guarenteed
    to be the ReLU output
    """

    def __init__(self, shared_alloc,
                 running_mean, running_var,
                 training=True, momentum=0.1, eps=1e-5):
        self.shared_alloc = shared_alloc
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, bn_weight, bn_bias, *inputs):

        # Create tensors that use shared allocations
        # One for the concatenation output (bn_input)
        # One for the ReLU output (relu_output)
        all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in all_num_channels[1:]:
            size[1] += num_channels
        with torch.no_grad():
            bn_input = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
            bn_output = F.batch_norm(bn_input, self.running_mean, self.running_var,
                                     bn_weight, bn_bias, training=self.training,
                                     momentum=self.momentum, eps=self.eps)
            assert self.shared_alloc.size() >= reduce(mul, size, 1)
            relu_output = inputs[0].new(self.shared_alloc).resize_(size)
            # Do ReLU - and have the output be in the intermediate storage
            torch.clamp(bn_output, min=0, out=relu_output)
        self.save_for_backward(*inputs)
        self.bn_weight = bn_weight
        self.bn_bias = bn_bias
        return relu_output

    def prepare_backward(self):
        inputs = self.saved_tensors
        all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in all_num_channels[1:]:
            size[1] += num_channels
        bn_input = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0].detach()
        # make bn_input requires_grad == True, must detach it from inputs first 
        self.bn_input = bn_input.requires_grad_()
        self.bn_weight = self.bn_weight.detach().requires_grad_()
        self.bn_bias = self.bn_bias.detach().requires_grad_()
        with torch.enable_grad():
            # Do batch norm
            self.bn_output = F.batch_norm(self.bn_input, self.running_mean, self.running_var,
                                          self.bn_weight, self.bn_bias, training=self.training,
                                          momentum=0, eps=self.eps)
        assert self.shared_alloc.size() >= reduce(mul, size, 1)
        # Do ReLU
        self.relu_output = inputs[0].new(self.shared_alloc).resize_(size)
        torch.clamp(self.bn_output, min=0, out=self.relu_output)

    def backward(self, grad_output):
        """
        Precondition: must call prepare_backward before calling backward
        """

        grads = [None] * (len(self.saved_tensors) + 2)
        inputs = self.saved_tensors

        # If we don't need gradients, don't run backwards
        if not any(self.needs_input_grad):
            return grads

        # BN weight/bias grad
        # With the shared allocations re-populated, compute ReLU/BN backward
        relu_grad_input = grad_output.masked_fill_(self.relu_output <= 0, 0)
        self.bn_output.backward(gradient=relu_grad_input)
        if self.needs_input_grad[0]:
            grads[0] = self.bn_weight.grad.data
        if self.needs_input_grad[1]:
            grads[1] = self.bn_bias.grad.data
        # Input grad (if needed)
        # Run backwards through the concatenation operation
        if any(self.needs_input_grad[2:]):
            all_num_channels = [input.size(1) for input in inputs]
            index = 0
            for i, num_channels in enumerate(all_num_channels):
                new_index = num_channels + index
                grads[2 + i] = self.bn_input.grad.data[:, index:new_index]
                index = new_index
        # remove intermediate variables
        del self.relu_output
        del self.bn_output
        del self.bn_input

        return tuple(grads)


class _DummyBackwardHook(Function):
    """
    A dummy function, which is just designed to run a backward hook
    This allows us to re-populate the shared storages before running the backward
    pass on the bottleneck layer
    The function itself is just an identity function
    """

    @staticmethod
    def forward(ctx, input, fn):
        ctx.fn = fn
        return input.data

    @staticmethod
    def backward(ctx, grad_output):
        ctx.fn.prepare_backward()
        return grad_output.data, None
