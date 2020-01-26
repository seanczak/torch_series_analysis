'''basic timeseries operations in pytorch'''

from numbers import Number

import torch 
import torch as tr

def causal_conv_1d( input, kernel, bias=None, stride=1, dilation=1, groups=1, special=False ):
    '''
    Causal Convolutions
    -------------------

    Implements the canonical time-series causal convolution formula:
    
    For q in Pout
        output[ t, :N, q] = \sum_{p in Pin} \sum_{l\in L} input[ t - l * dilation, :N, p ] kernel[ l, p, q ]
    
    Here:
        l = 0 .... L-1  -- most recent to most distant filter values
        
    This is the regular interpretation of a convolution in time-series 
    and is different from torch which implements the computer-vision
    definition of convolution as a centered (i.e. non-causal) correlation.
        
    Uses torch.nn.functional.conv1d under the hood.


    Parameters
    ----------
    input:
        (3 tensor) T x N x Pin or (2 tensor) T x N (assumes Pin = 1)
    kernel:
        (3 tensor) L x Pin x Pout or (2 tensor) L x Pout or (1 tensor)  L
    bias:
        optional bias of shape Pout. Default: None
    stride:
        the stride of the convolving kernel. Can be a single number or a one-element tuple (sW,). Default: 1
        Keep this 1 - other values result in downsampling the output.
    dilation:
        the spacing between kernel elements. Can be a single number or a one-element tuple (dW,). Default: 1
    groups:
        split input into groups, Pin and Pout should be divisible by the number of groups. Default: 1
        groups controls the connections between inputs and outputs.
        Pin and Pout must both be divisible by groups. For example,
        * At groups=1, all inputs are convolved to all outputs.
        
        * At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half
           the input channels, and producing half the output channels, and both subsequently concatenated.
        
        * At groups= in_channels, each input channel is convolved with its own set of filters,
            of size floor( Pout/ Pin )
    special:
        False: (default)
            Conv. kernel is not run on each feature independently.
            All the Pin input channels are merged together to produce Pout channels.
            The Pin of `input` needs to match `Pin` of kernel and `output` has `Pout` channels.
        True:
            This mimics the ability to run the same convolution kernel on each feature independently !!
            If the Pin of the kernel is 1,  but Pin of input is not 1:
                Then input is reshaped to   T x ( N.Pin ) x 1 (i.e. flattened out)
                The filter is applied independently to each input feature
                The output is then reshaped to T x N x Pin x Pout

    Returns
    -------
    output  : tensor of shape  T x N x Pout       (if special=False)
                               T x N x Pin x Pout (if special=True)
    

    '''
    if input.ndim == 2:
        # Expand dims and set Pin = 1
        input = input.unsqueeze( -1 )
    if kernel.ndim == 1:
        # Expand dims and set Pout = 1
        kernel = kernel.unsqueeze( -1 )
    if kernel.ndim == 2:
        # Expand dims and set Pin = 1
        kernel = kernel.unsqueeze( 1 )
    L, kPin, Pout = kernel.shape
    T, N, xPin = input.shape
    # -- special handling --
    if special:
        if kPin != 1:
            raise ValueError( 'kernel must have 1-channel input for special mode to make sense' )
        # reshape input to be T x (N.Pin) x 1. This will allow running kernel on each feature independently
        input = input.reshape( T, N * xPin ).unsqueeze( -1 )
    elif kPin != xPin:
        raise ValueError( f'kernel input channels {kPin} not compatible wtih X input-channes {xPin}' )
    if bias is not None:
        if bias.ndim != 1 and bias.shape[0] != Pout:
            raise ValueError( 'bias is not of proper shape - should be (Pout, )' )
    # transpose T x N x Pin input to N x Pin x T as needed by pytorch
    X = input.transpose( 0, 1 ).transpose( 1, 2 )
    # transpose L x Pin x Pout kernel to be  Pout x Pin x L - followed by a flip to use `correlation` as the backend
    W = kernel.transpose( 0, 2 ).flip( -1 )
    # Note the padding adjustment for causality - with dilation
    padding = ((L - 1) * dilation,)
    # this is actually a convolution. Returns output of shape N x Pout x (T + padding + 1)
    Y = tr.nn.functional.conv1d( X, W, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups )
    # reshape to be T x N x Pout and truncate off padded values
    output = Y.transpose( 1, 2 ).transpose( 0, 1 )[:T]
    # --- special handling --
    if special:
        # we had collapses some dimensions - lets fix that here
        output = output.reshape( T, N, xPin, Pout )
    return output


class CausalConv1D( torch.nn.Module ):
    '''
    Causal convolutions
    -------------------------------------
    (Module wrapper around causal_conv_1d)
    
    Parameters
    ----------
    L: kernel lenght
    Pin: input channels
    Pout: output channels
    stide:
        See tr.nn.Conv1d
    dilation:
        See tr.nn.Conv1d
    groups:
        See tr.nn.Conv1d
    bias:
        if true, adds a bias term
    special:
        False: (default)
            Conv. kernel is not run on each feature independently.
            All the Pin input channels are merged together to produce Pout channels.
            The Pin of `input` needs to match `Pin` of kernel and `output` has `Pout` channels.
        True:
            This mimics the ability to run the same convolution kernel on each feature independently !!
            If the Pin of the kernel is 1,  but Pin of input is not 1:
                Then input is reshaped to   T x ( N.Pin ) x 1 (i.e. flattened out)
                The filter is applied independently to each input feature
                The output is then reshaped to T x N x Pin x Pout
    dropout: float
        dropout percentage
        if None - no dropout

    
    Attributes:
    ----------
        kernel (Tensor): the learnable kernels of the module of shape L x Pin x Pout.
            default initialization is kaiming_uniform
        bias (Tensor):   The learnable bias of the module of shape Pout.
                Initialized from xavier uniform.
                
    Usage:
    -----
        output = conv( input )  # regular convolution
    '''
    __doc__ += causal_conv_1d.__doc__
    
    def __init__( self, L, Pin, Pout, stride=1, dilation=1, groups=1, bias=True, special=False ):
        super().__init__()
        if Pin % groups != 0:
            raise ValueError( 'Pin must be divisible by groups' )
        if Pout % groups != 0:
            raise ValueError( 'Pout must be divisible by groups' )
        self.Pin = Pin
        self.Pout = Pout
        self.L = L
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.special = special
        self.kernel = tr.nn.Parameter( tr.Tensor( L, Pin // groups, Pout ) )
        if bias:
            self.bias = tr.nn.Parameter( tr.Tensor( Pout ) )
        else:
            self.register_parameter( 'bias', None )
        self.dropout = tr.nn.Dropout( p=dropout ) if dropout is not None else lambda x: x
        self.reset_parameters()
    
    def forward( self, input ):
        output = causal_conv_1d( input, kernel=self.dropout( self.kernel ), bias=self.bias, stride=self.stride,
                                 dilation=self.dilation, groups=self.groups, special=self.special )
        return output

