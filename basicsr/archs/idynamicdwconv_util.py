from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn as nn

from collections import namedtuple
from string import Template

# -----------------------------------------------------------------------------
# CuPy 是 GPU 用的 JIT kernel 編譯器。Pi/CPU-only 環境裝不起來,包成 try/except.
# 所有 cupy 相依路徑(@cupy._util.memoize、cupy.RawModule、Stream(cuda_stream))
# 都用 _HAS_CUPY 守衛起來;CPU 環境會走 _idynamic_cuda 裡的 PyTorch fallback.
# -----------------------------------------------------------------------------
try:
    import cupy
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


# -----------------------------------------------------------------------------
# load_kernel: JIT 編譯 CUDA C 字串.
# CuPy 不存在時提供 stub,import 階段不會 NameError;真的被呼叫到時才 raise.
# 因為 CPU fallback 走 _idynamic_cuda 的另一條 branch,這個 stub 不會被觸發.
# -----------------------------------------------------------------------------
if _HAS_CUPY:
    @cupy._util.memoize(for_each_device=True)
    def load_kernel(kernel_name, code, **kwargs):
        code = Template(code).substitute(**kwargs)
        kernel_code = cupy.RawModule(code=code)
        return kernel_code.get_function(kernel_name)
else:
    def load_kernel(kernel_name, code, **kwargs):
        raise RuntimeError(
            "load_kernel requires CuPy/CUDA; this environment is CPU-only. "
            "If you reached this error, the CPU fallback in _idynamic_cuda "
            "did not catch your code path."
        )


_idynamic_kernel = '''
extern "C"
__global__ void idynamic_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          const int weight_offset = (((((n * ${groups} + g) * ${kernel_h}) + kh) * ${kernel_w} + kw) * ${top_height} + h)
            * ${top_width} + w;
          value += weight_data[weight_offset] * bottom_data[offset];
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_idynamic_kernel_backward_grad_input = '''
extern "C"
__global__ void idynamic_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            const int weight_offset = (((((n * ${groups} + g) * ${kernel_h}) + kh) * ${kernel_w} + kw) * ${top_height} + h_out)
            * ${top_width} + w_out;
            value += weight_data[weight_offset] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_idynamic_kernel_backward_grad_weight = '''
extern "C"
__global__ void idynamic_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${num};
      ${Dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
              * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
              * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''


CUDA_NUM_THREADS = 512
# if you use in 3090 and above, please set 1024 for the fastest calculation
def GET_BLOCKS(N, NUM_THREADS=CUDA_NUM_THREADS):
    return (N + NUM_THREADS - 1) // NUM_THREADS


class _idynamic(Function):
    """
    GPU-only autograd Function backed by hand-written CUDA kernels via CuPy.
    On CPU, _idynamic_cuda below routes around this class entirely (uses the
    PyTorch fallback in pft_cpu_ops.idynamic_conv), so this class only needs
    to be importable -- its forward/backward will never run on CPU.
    """
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)

        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('idynamic_forward_kernel', _idynamic_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, groups=weight.size()[1],
                            bottom_height=height, bottom_width=width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels, groups=weight.size()[1],
                   bottom_height=height, bottom_width=width,
                   top_height=output_h, top_width=output_w,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('idynamic_backward_grad_input_kernel',
                                _idynamic_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())

                n = grad_weight.numel()
                opt['nthreads'] = n

                f = load_kernel('idynamic_backward_grad_weight_kernel',
                                _idynamic_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_weight, None, None, None


def _idynamic_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """idynamic kernel - CUDA 上用自訂 kernel,CPU 上用 PyTorch fallback."""
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // _pair(stride)[0] == weight.size(-2)
    assert input.size(-1) // _pair(stride)[1] == weight.size(-1)
    if input.is_cuda and _HAS_CUPY:
        out = _idynamic.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out = out + bias.view(1, -1, 1, 1)
        return out
    # CPU / non-CUPY 環境:純 PyTorch 實作
    from .sfmformer_cpu_ops import idynamic_conv
    return idynamic_conv(input, weight, bias=bias,
                         stride=stride, padding=padding, dilation=dilation)


class IDynamicDWConv(nn.Module):
    """
        IDynamicDWConv: HyperNet for the weight of DynamicDWConv
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 group_channels, bias=True):
        super(IDynamicDWConv, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=bias),
            nn.Conv2d(channels // reduction_ratio, channels // reduction_ratio, kernel_size=kernel_size,
                      padding=kernel_size//2, groups=channels//reduction_ratio, bias=bias),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // reduction_ratio, kernel_size ** 2 * self.groups, 1, bias=bias)
        )

    def forward(self, x):
        weight = self.conv2(self.conv1(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _idynamic_cuda(x, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        return out


class IDynamic(nn.Module):
    """
        IDynamicDWConv: HyperNet for the weight of DynamicDWConv
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 group_channels, bias=True):
        super(IDynamic, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        reduction_ratio = 8
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=bias),
            nn.Conv2d(channels // reduction_ratio, channels // reduction_ratio, kernel_size=kernel_size,
                      padding=kernel_size//2, groups=channels//reduction_ratio, bias=bias),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // reduction_ratio, kernel_size ** 2 * self.groups, 1, bias=bias)
        )

    def forward(self, x_main, x):
        weight = self.conv2(self.conv1(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _idynamic_cuda(x_main, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        return out
