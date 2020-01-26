'''
ARMA operation in pytorch
'''

import torch as tr


class ARMA( tr.nn.Module ):
    '''
    ARMA Module
    -----------
    
    Solves:
        Y_t = \sum_p A_p Y_{t-p-1}  + \sum_q B_q X_{t-q}
        
        time -> 0-th dim
        batch -> 1-st dim
        features -> 2-nd dim
        
  
    Parameters
    ----------
    num_inputs
        1 or more. Not needed for univariate
    num_outputs
        1 or more.  Not needed for univariate.
    P:
        AR order
    Q
        MA order
    style:
        ARMA.vector, ARMA.mass_univariate, ARMA.univariate
        vector => fully vector AR and MA processes (i.e. dense weights matrices)
        mass_univariate => mass univariate AR and MA (i.e. diagonal weights matrices)
        univariate => purely univariate AR and MA (i.e. scalar weights)
        
    dropout: float
        dropout percentage
        if None - no dropout
    '''
    
    vector = 'vector'
    mass_univariate = 'mass_univariate'
    univariate = 'univariate'
    
    def __init__( self, num_inputs=1, num_outputs=1, P=1, Q=1, style='mass_univariate', dropout=False ):
        super().__init__()
        # sanity test the arguments.
        if style == ARMA.mass_univariate:
            assert num_inputs == num_outputs, 'input and output must be same size for mass_univariate'
        # just force this ... fuck it.
        elif style == ARMA.univariate:
            num_inputs = num_outputs = 1
        elif style == ARMA.vector:
            assert num_inputs >= 1 and num_outputs >= 1, 'whaaa ??'
        else:
            raise ValueError( 'invalid style {style}' )
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.style = style
        self.P = P
        self.Q = Q
        self.style = style
        #  the auto-regressive weights
        self.ar_weights = []
        for p in range( P ):
            # multivariate
            if style == ARMA.vector:
                Ap = tr.nn.Parameter( tr.empty( num_outputs, num_outputs ) )
            # univariate or mass univariate
            else:
                Ap = tr.nn.Parameter( tr.empty( num_outputs ) )
            self.ar_weights.append( Ap )
            self.register_parameter( f'A_{p}', Ap )
        #  the moving-average weights
        self.ma_weights = []
        for q in range( Q ):
            # multivariate
            if style == ARMA.vector:
                Bq = tr.nn.Parameter( tr.empty( num_inputs, num_outputs ) )
            # univariate or mass univariate
            else:
                Bq = tr.nn.Parameter( tr.empty( 1, num_inputs ) )
            self.ma_weights.append( Bq )
            self.register_parameter( f'B_{q}', Bq )
        self.dropout = tr.nn.Dropout( p=dropout ) if dropout is not None else lambda x: x

    
    def expected_num_outputs( self, X ):
        return X.shape[-1] if self.style in (ARMA.mass_univariate, ARMA.univariate) else self.num_outputs
    
    def check_input( self, X, Yo ):
        '''check input correctness'''
        if X.ndim != 3 or Yo.ndim != 3:
            raise TypeError( f'incorrect number of dims for X ({X.ndim}) and Yo ({Yo.ndim}), expected 3' )
        if self.style in (ARMA.mass_univariate, ARMA.vector) and self.num_inputs != X.shape[-1]:
            raise TypeError( f'Incorrect size for X on dim=2 ({X.shape[-1]}) expected {self.num_inputs}' )
        num_outputs = self.expected_num_outputs( X )
        if Yo.shape != (self.P, X.shape[1], num_outputs):
            raise TypeError( f'Incorrect size for Yo ({Yo.shape}) expected {(self.P, X.shape[1], num_outputs)}' )
    
    def forward( self, X, times=None, state=None ):
        '''
        forward pass
        
        Parameters
        ----------
        X
        state:
            Teh state / buffers returned from a previous call to ARMA.
            Contains the initial value of Y of shape (Q, batch, num_outputs)
        times:
            ignored

        Returns
        -------

        '''
        # expected output size
        T, batch_size, num_outputs = X.shape[0], X.shape[1], self.expected_num_outputs( X )
        # determine the initial state of y
        if state is None:
            state = { }
        Yo = state.get( 'Yo', tr.zeros( self.P, batch_size, num_outputs, dtype=X.dtype, device=X.device ) )
        # test shape correctness
        self.check_input( X, Yo )
        # TODO make the input nan-safe
        # TODO X = nan_safe(X)
        # convert the priming Yo's into a list of Yo's along with the future Y'
        Ys = [Yo[p] for p in range( self.P )] \
             + [tr.zeros( 1, batch_size, num_outputs, dtype=X.dtype, device=X.device ) for _ in range( T )]
        if self.style == ARMA.vector:
            Ys = ARMA._varma( X, Ys, T=T, P=self.P, Q=self.Q, ar_weights=self.ar_weights, ma_weights=self.ma_weights )
        else:
            Ys = ARMA._arma( X, Ys, T=T, P=self.P, Q=self.Q, ar_weights=self.ar_weights, ma_weights=self.ma_weights )
        Y = tr.cat( Ys[self.P:], dim=0 )
        # save for the next round
        state['Yo'] = Y[-self.P:]
        return Y, state

    @staticmethod
    def _varma( X, Ys, T, P, Q, ar_weights, ma_weights ):
        '''vector ARMA'''
        for t in range( T ):
            for p in range( P ):
                Ys[P + t] += Ys[P + t - p - 1] @ self.dropout( ar_weights[p] )
            for q in range( min( t + 1, Q ) ):
                Ys[P + t] += X[t - q] @ self.dropout( ma_weights[q] )
        return Ys

    @staticmethod
    def _arma( X, Ys, T, P, Q, ar_weights, ma_weights ):
        '''(mass) univariate ARMA'''
        for t in range( T ):
            for p in range( P ):
                Ys[P + t] += Ys[P + t - p - 1] * self.dropout( ar_weights[p] )
            for q in range( min( t + 1, Q ) ):
                Ys[P + t] += X[t - q] * self.dropout( ma_weights[q] )
        return Ys
