
import torch as tr 
from causal_conv import causal_conv_1d

class Wavelet( tr.nn.Module ):
    '''
    Causal Wavelet implementation (i.e. a stack of dilated convolution)
    
    Set up like this:
        
    out_1 = causal_conv( X, L=L, dilation=1 )
    out_2 = causal_conv( out_1, L=L, dilation=2 )
    out_3 = causal_conv( out_2, L=L, dilation=4 )
    ...
    out_n = causal_conv( out_2, L=L, dilation=2^n )
    
    output = [out_1 ... out_n]

    Parameters
    ----------
    L:
        length of the kernel at each layer
    Pin:
        total number of input channels. If Pin = 1 - then the special flag has meaning.
    Pout
        The number of output channels per dilation layer !
    num_layers
        number of layers
    dilation
        default 2
    special
        if True, then each input feature is processed independently with the same kernel.
                 Pin should be 1 for this.
        if False, then the input features are linearly combined in each dilation stage
        See
    dropout: float
        dropout percentage
        if None - no dropout
    
    Input
    -----
        T x N x Pin  or T x N (sets Pin = 1)
        
    Output:
    ------
    if special = True
        T x N x Pin x Pout x num_layers
    if special = False:
        T x N x Pout x num_layers
    
    This is not a wavenet. Wavenet Reference is here
    -------------------------------------------------
    https://github.com/NVIDIA/nv-wavenet/blob/master/pytorch/wavenet.py
    
    '''
    
    def __init__( self, L, Pin, Pout, num_layers=2, dilation=2, special=False, dropout=None ):
        '''
        '''
        super().__init__()
        
        self.special = special
        if special and Pin > 1:
            raise ValueError( 'Pin must be 1 for special mode to make sense' )
        self.L = L
        self.Pin = Pin
        self.Pout = Pout
        self.num_layers = num_layers
        self.dilation = dilation
        self.dropout = tr.nn.Dropout( p=dropout ) if dropout is not None else lambda x: x
        # build the convolution kernels
        self._kernels = []
        for l in range( self.num_layers ):
            self._kernels.append( tr.nn.Parameter( tr.Tensor( L, Pin, Pout ) ) )
            self.register_parameter( f'kernel_{l}', self._kernels[l] )
            Pin = Pout
        self.reset_parameters()
    
    def forward( self, input ):
        if input.ndim == 2:
            # Expand dims and set Pin = 1
            input = input.unsqueeze( -1 )
        if input.ndim != 3:
            raise ValueError( 'input must be a 3 tensor' )
        T, N, xPin = input.shape
        # special handling here
        if self.special:
            input = input.reshape( T, N * xPin ).unsqueeze( -1 )
        outputs = []
        for l in range( self.num_layers ):
            # Note - special handling is done by Wavelet - not conv_1d.
            output = causal_conv_1d( input, kernel=self.dropout( self._kernels[l] ), dilation=self.dilation ** l,
                                     special=False )
            input = output
            # undo any special handling here
            outputs.append( (output.reshape( T, N, xPin, self.Pout ) if self.special else output).unsqueeze( -1 ) )
        # concatenate all the dilated convolutions along the last dimension
        return tr.cat( outputs, dim=-1 )
