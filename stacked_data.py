import numpy as np
import torch


def np_to_torch(arr): 
    '''Prepare a numpy array for a pytorch model'''
    return torch.from_numpy(arr).float()


def running_view(arr, window, axis=-1):
    """
    return a running view of length 'window' over 'axis'
    the returned array has an extra last dimension, which spans the window
    """
    shape = list(arr.shape)
    shape[axis] -= (window-1)
    assert(shape[axis]>0)
    return np.lib.index_tricks.as_strided(
        arr,
        shape + [window],
        arr.strides + (arr.strides[axis],))



class stacked_univariant_data():
    '''
    Preps a dataset for a sequential model that is expecting stacked inputs. While there is no equivalence
    requirement between input and output length. Post-model visualization is cleaner when input_len = output_len

    Arguments:
        - data: numpy dataset should be in the shape [num_samples, sequence_length]
            NOTE: Must be a numpy array.
        - input_len: number of floats passed at each timestep to a model
        - output_len: number of floats to be predicted at each timestep
        - tau_offset: number of points skipped between end of input and start of output

    Attributes:
        - x: torch.Tensor of inputs into the model of shape [num_samples, *, input_len] where * is the 
            properly shaped time dimension
        - y: torch.Tensor of targets of the model of shape [num_samples, *, input_len] where * is the
            properly shaped time dimension
        - tx and ty: Corresponding time indices for the inputs and outputs respectively. Included to simplify
            visualizations
    '''

    def __init__(self,data, input_len=1, output_len=1, tau_offset = 0):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.tau_offset = tau_offset
        
        # create time array which is just indices
        self.time = np.arange(data.shape[1])
        # heavy lifting is done by this function below
        self.shape_ins_outs()
   
    
    def shape_ins_outs(self):
        '''Utilizes `running_view` above to create a 3D tensor based on model's input, output, tau specs'''

        i, o, tau = self.input_len, self.output_len, self.tau_offset
        combined_len = i + o + tau

        # create the datasets by taking every i'th row (if i is input_len) so inputs don't overlap
        data = running_view(self.data,combined_len)[:,::i]
        time = running_view(self.time,combined_len)[::i] # time doesn't have batch dim

        # split each row into input and output, since each row is length of the sum of the two, its just slicing
        x = data[:,:,:i]
        y = data[:,:,(i + tau):]

        # convert to torch
        self.x, self.y = np_to_torch(x), np_to_torch(y)
        
        # times that correspond with each example (note that its only 2d since each example is same length)
        self.tx = time[:,:i] # again, time doesn't have batch dim
        self.ty = time[:,(i + tau):]



class stacked_multivariant_data():
    '''
    Preps a dataset for a sequential model that is expecting stacked inputs. While there is no equivalence
    requirement between input and output length. Post-model visualization is cleaner when input_len = output_len

    Arguments Taken:
        - data: numpy dataset should be in the shape [num_samples, sequence_length, num_channels]
            NOTE: Must be a numpy array.
        - input_len: number of floats passed by each channel at each timestep to a model
            NOTE: The actual input into the model will be of size (input_len * num_channels)
        - output_len: number of floats to be predicted for each channel at each timestep.
            Similarly, the actual output of the model will be of size (output_len * num_channels)
        - tau_offset: number of points skipped between end of input and start of output

    Attributes Returned:
        - x: torch.Tensor of inputs into the model of shape [num_samples, *, input_len] where * is the 
            properly shaped time dimension
        - y: torch.Tensor of targets of the model of shape [num_samples, *, input_len] where * is the
            properly shaped time dimension
        - tx and ty: Corresponding time indices for the inputs and outputs respectively. Included to simplify
            visualizations. 
    '''

    def __init__(self, data, input_len=1,output_len=1,tau_offset = 0):

        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.tau_offset = tau_offset
        self.time = np.arange(data.shape[1])

        # create stacked_univariant_data instance for each channel
        self.channel_data = []
        num_channels = data.shape[2]
        for channel in range(num_channels):
            chan_data = stacked_univariant_data(data[:,:,channel], input_len=input_len,output_len=output_len,tau_offset = tau_offset,overlap = overlap)
            self.channel_data.append(chan_data)
      
        # using the above channels, concat inputs/outputs for model along input dimension
        self.x = torch.cat( [self.channel_data[i].x for i in range(num_channels)] , 2) # concat inputs along 3rd dim
        self.y = torch.cat( [self.channel_data[i].y for i in range(num_channels)] , 2) # concat outputs along 3rd dim

        # only need one time vector
        self.tx = self.channel_data[0].tx
        self.ty = self.channel_data[0].ty