'''forward hooks'''

from collections import defaultdict

import pandas as pd
import torch as tr
from torch.utils import tensorboard as tb
import os


class Tensorboard:
    '''
    Provides method to monitor
        (inputs, outputs, parameters) x (values, gradients)
    for
        (tr.Tensor, tr.nn.Module, tr.nn.Parameters, flow through a torch graph )
        
    
    Example
    -------
        first install tensorboard "pip install tensorboard"
        Then go to the log_dir below and startup tensorboard from the command line.
        >> tensorboard --logdir <path_to_logs>
        
        ```python
        import torch as tr
        
        from tensorboard as mxtl; reload( mxtl )
        
        # optional - set tensorboard logging dir, flush freq
        # Tensorboard.log_dir = <default>
        # Tensorboard.flush_secs = 30
        
        # set the logging patience from default 50 to 50
        mxtl.Tensorboard.patience = 50
        
        # Insert a monitor module into the graph to log flow through it to tensorboard
        model = tr.nn.Sequential(
            tr.nn.Linear( in_features=3, out_features=5)  ,
            mxtl.Tensorboard.monitor_flow( name='linear_1'),
            tr.nn.Linear( in_features=5, out_features=1 )  ,
            mxtl.Tensorboard.monitor_flow( name='linear_2'),
        )
        
        #A monitor which instruments / inspects the inputs and outputs of the wrapped Module
        w = tr.tensor( np.ones(3), requires_grad=True, dtype=tr.float )
        w_monitor = mxtl.Tensorboard.monitor( name='weight_1', node=w )
        
        
        # Monitor all the parameters of model
        mxtl.Tensorboard.monitor_parameters( name='model', module=model )
        
        for _ in range( 1000 ):
            X = tr.as_tensor( randn( 20, 3 ), dtype=tr.float )
            # common branch
            Z1 =  (X @ w).reshape( -1, 1)
            Z2 = model( X )
            # two split branches
            Y1 = Z1 ** 2
            Y2 =  Z2 ** 3
            L =  Y1.sum() + Y2.sum()
            L.backward()
        
        # stuff is logged to tensorboard. Also, examine logs as dataframe
        model[1].logs
        w_monitor.logs 
        
        ```
    '''
    # configuration for the tensorboard module
    cwd = os.getcwd()
    log_dir = f'{cwd}/tmp/tensorboard'
    flush_secs = 30
    patience = 50
    
    class monitor:
        '''
        Instruments a tr.Tensor or tr.nn.Module to monitor inputs, outputs and and gradients and log to tensorboard.
        NOTE: not all features are available when instrumenting tr.Tensors (only output value and input gradients)
        
        Parameters
        ----------
        name:
            The name for this monitor
        node:
            The node (tr.Tensor or tr.nn.Module) to log
        inputs
            if True, logs input (values and gradients)
        outputs
            if True, logs output (values  and gradients)
        parameters:
            if True, logs all parameters registerd with tr.nn.Module (gradients and values)
        values:
            if True, logs all the values of (inputs, outputs and gradients)
        gradients
            if True, logs backprop gradient (inputs, outputs and parameters)
            
        Returns
        --------
            Tensorboardlogger object
    
        '''
        
        def __init__( self, name, node, parameters=False, inputs=True, outputs=True, gradients=True, values=True ):
            self.node = node
            self.name = name
            self.parameters = parameters
            self.inputs = inputs
            self.outputs = outputs
            self.values = values
            self.gradients = gradients
            # counter for forward and backward passes
            self._forward_it = 0
            self._backward_it = 0
            # the tensorboard monitor
            self._logger = tb.writer.SummaryWriter( log_dir=Tensorboard.log_dir, flush_secs=Tensorboard.flush_secs )
            # store as _logs for later examination
            self._logs = []
            
            # -- register the hooks ---
            # register backward hooks for tr.Tensor - no forward hook
            if isinstance( self.node, tr.Tensor ):
                
                def backward_hook( grad ):
                    ''' the backward hook  - for logging gradients'''
                    self._backward_it += 1
                    # log tensor output value
                    if self.values and self.outputs:
                        self._log_tensors( it=self._backward_it, name=f'{self.name}/outputs/values', tensors=self.node )
                    # log gradients wrt input if requested
                    if self.gradients and self.inputs:
                        self._log_tensors( it=self._backward_it, name=f'{self.name}/inputs/gradients', tensors=grad )
                    return
                
                self.node.register_hook( backward_hook )
            
            # register forward and backward hooks for a module
            elif isinstance( self.node, tr.nn.Module ):
                
                def backward_hook( module, grad_input, grad_output ):
                    ''' the backward hook  - for logging gradients'''
                    self._backward_it += 1
                    # log gradients only if requested
                    if not self.gradients:
                        return
                    # log gradients with respect to all module outputs
                    if self.outputs:
                        self._log_tensors( it=self._backward_it, name=f'{self.name}/outputs/gradients',
                                           tensors=grad_output )
                    # log gradients wtih repsect to all module inputs
                    if self.inputs:
                        self._log_tensors( it=self._backward_it, name=f'{self.name}/inputs/gradients/',
                                           tensors=grad_input )
                    # log gradients with repsect to all module parameters
                    if self.parameters:
                        self._log_tensors(
                            it=self._backward_it, name=f'{self.name}/parameters/gradients',
                            tensors={ name: param.grad for name, param in self.node.named_parameters() }
                        )
                
                def forward_hook( module, input, output ):
                    self._forward_it += 1
                    # log values only if requested
                    if not self.values:
                        return
                    # log inputs if requested
                    if self.inputs:
                        self._log_tensors( it=self._forward_it, name=f'{self.name}/inputs/values', tensors=input )
                    # log outputs if requested
                    if self.outputs:
                        self._log_tensors( it=self._forward_it, name=f'{self.name}/outputs/values', tensors=output )
                    # log parameters if requested
                    if self.parameters:
                        self._log_tensors( it=self._forward_it, name=f'{self.name}/parameters/values',
                                           tensors=dict( self.node.named_parameters() ) )
                
                node.register_backward_hook( backward_hook )
                node.register_forward_hook( forward_hook )
        
        def _log_tensors( self, it, name, tensors ):
            '''log a tensor or a list / tuple of tensors '''
            # log only every patience rounds
            if (it % Tensorboard.patience) != 0:
                return
            if tensors is None:
                return
            elif isinstance( tensors, tr.Tensor ):
                tensors = { 'tensor_0': tensors }
            elif isinstance( tensors, list ) or isinstance( tensors, tuple ):
                tensors = [tensor for tensor in tensors if tensor is not None]
                tensors = { f'tensor_{i}': tensor for i, tensor in enumerate( tensors ) }
            assert isinstance( tensors, dict ), f'invalid type = {type(tensors)}'
            for tensor_name, tensor in tensors.items():
                if tensor is None:
                    continue
                # add the tensor norm
                norm = tensor.data.cpu().norm().item()
                self._logger.add_scalar( f'{name}/{tensor_name}/norm', norm, it )
                # variance along 0-th dim
                var = tensor.data.cpu().numpy().reshape( len( tensor ), -1 ).var( axis=0 )
                var_dict = { str( i ): v for i, v in enumerate( var ) }
                self._logger.add_scalars( f'{name}/{tensor_name}/batch_var', var_dict, it )
                # add tensor historgram
                self._logger.add_histogram( f'{name}/{tensor_name}/hist', tensor.data.cpu(), it )
                # save for later examination
                self._logs.append( dict( it=it, norm=norm, batch_var=var, shape=tensor.shape, name=name,
                                         tensor=tensor_name ) )
        
        @property
        def logs( self ):
            '''
            display the logs as dataframes generated for later analysis
            '''
            return pd.DataFrame( self._logs ).set_index( ['tensor', 'name', 'it'] ).sort_index()
    
    @staticmethod
    def monitor_parameters( name, module ):
        '''
        A method that builds a monitor to report on all the parameters in a tr.nn.Module.
        '''
        assert isinstance( module, tr.nn.Module ), f'cannot monitor params with type(module) = {type(module)}'
        return Tensorboard.monitor( name=name, node=module, inputs=False, outputs=False, parameters=True,
                                    gradients=True, values=True )
    
    
    class monitor_flow( tr.nn.Module ):
        '''
        A passthrough module that you can insert into your graph to monitor the flow through a particular stage of
        the graph, i.e. activations going forward and gradients propagating backward through this edge..
        
        Parameters
        ----------
        name
            name of the monitor (for display in tensorboard)
        patience
            After how many forward/backward passes to log.
        log_dir
        
        flush_secs
            delay for writing out to tensorboard
        
       
        '''
        # global counter for all monitors
        count = 0
        
        def __init__( self, name=f'monitor_{count}' ):
            super().__init__()
            self.name = name
            Tensorboard.monitor_flow.count += 1
            # register tensorboard hooks to monitor the gradients and output of this module
            self.tblogger = Tensorboard.monitor( name=name, node=self, inputs=False, parameters=False,
                                                 outputs=True, gradients=True, values=True )
        
        def forward( self, input ):
            '''simple pass through'''
            return input
        
        @property
        def logs( self ):
            '''
            display the logs as dataframes generated for later analysis
            '''
            return self.tblogger.logs