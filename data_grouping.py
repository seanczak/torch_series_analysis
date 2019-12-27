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


class univariate_data():
    '''
    Prepares a pytorch tensor from a numpy [m x n] matrix of m univariate data sequences of length n 
    for time series prototyping in pytorch.

    Arguments Taken:
        - data: numpy dataset should be in the shape [num_samples, sequence_length]
            NOTE: Must be a numpy array.
        - input_len: `Default = 1`
            number of floats passed by each channel to a model.
        - output_len: `Default = 1`
            number of floats to be predicted for each channel.
        - tau_offset: `Default = 0`
            number of points skipped between end of input and start of output
        - inputs_overlap: `Default = False`
            Preps what I call a "segmented" dataset because it will break a sequence into many parts if necessary
            but each piece does not carry information to the other pieces in the model's implementation, 
            thus the inputs can overlap. Examples of these types of models are ARp models. 
            Eg:
                Given [0,1,2,3,4...] and inputs_overlap = True yields this matrix of inputs:
                    [[0,1]
                     [1,2]
                     [2,3]
                     [3,4]
                      ...]
                whereas inputs_overlap = False yields:
                    [[0,1]
                     [2,3]
                      ...]

    Attributes Returned:
        - x: torch.Tensor of inputs into the model of shape [num_samples, *, input_len] where * is the 
            properly shaped time dimension (close but not always seq_len/input_len)
        - y: torch.Tensor of targets of the model of shape [num_samples, *, input_len] where * is the
            properly shaped time dimension
        - tx and ty: Corresponding time indices for the inputs and outputs respectively. Included to simplify
            visualizations. Shape [*, input_len] or  [*, output_len] respectively where * is same as x, y tensor
    '''

    def __init__(self,data, input_len=1, output_len=1, tau_offset = 0, inputs_overlap=False):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.tau_offset = tau_offset
        
        # create time array which is just indices
        self.time = np.arange(data.shape[1])
        # heavy lifting is done by this function below
        self.shape_ins_outs(inputs_overlap)
   
    def shape_ins_outs(self,inputs_overlap):
        '''Utilizes `running_view` above to create a 3D tensor based on model's input, output, tau specs'''

        i, o, tau = self.input_len, self.output_len, self.tau_offset
        combined_len = i + o + tau

        # create the datasets
        if inputs_overlap:
            # inputs overlap
            data = running_view(self.data,combined_len)
            time = running_view(self.time,combined_len) # time doesn't have batch dim
        else:
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


class multivariate_data():
    '''
    Prepares a pytorch tensor from a numpy [m x n x k] matrix of m multivariate data sequences of length n and k channels 
    for time series prototyping in pytorch.

    Arguments Taken:
        - data: numpy dataset should be in the shape [num_samples, sequence_length, num_channels]
            NOTE: Must be a numpy array.
        - input_len: `Default = 1`
            number of floats passed by each channel to a model.
            NOTE: The actual input into the model will be of size (input_len * num_channels)
        - output_len: `Default = 1`
            number of floats to be predicted for each channel. The actual output of the model 
            will be of size (output_len * num_channels)
        - tau_offset: `Default = 0`
            number of points skipped between end of input and start of output
        - inputs_overlap: `Default = False`
            Preps what I call a "segmented" dataset because it will break a sequence into many parts if necessary
            but each piece does not carry information to the other pieces in the model's implementation, 
            thus the inputs can overlap. Examples of these types of models are ARp models. 
            Eg:
                Given [0,1,2,3,4...] and inputs_overlap = True yields this matrix of inputs:
                    [[0,1]
                     [1,2]
                     [2,3]
                     [3,4]
                      ...]
                whereas inputs_overlap = False yields:
                    [[0,1]
                     [2,3]
                      ...]
            

    Attributes Returned:
        - x: torch.Tensor of inputs into the model of shape [num_samples, *, num_channels*input_len] 
            where * is the properly shaped time dimension (close but not always seq_len/input_len)
        - y: torch.Tensor of targets of the model of shape [num_samples, *, num_channels*input_len] 
            where * is the properly shaped time dimension
        - tx and ty: Corresponding time indices for the inputs and outputs respectively. Included to simplify
            visualizations. Shape [*, input_len] or  [*, output_len] respectively where * denotes same as x, y tensor
    '''

    def __init__(self, data, input_len=1,output_len=1,tau_offset = 0, inputs_overlap=False):

        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.tau_offset = tau_offset
        self.time = np.arange(data.shape[1])

        # create univariate_data instance for each channel
        self.channel_data = []
        num_channels = data.shape[2]
        for channel in range(num_channels):
            chan_data = univariate_data(data[:,:,channel], input_len=input_len,output_len=output_len,tau_offset = tau_offset, inputs_overlap=inputs_overlap)
            self.channel_data.append(chan_data)
      
        # using the above channels, concat inputs/outputs for model along input dimension
        self.x = torch.cat( [self.channel_data[i].x for i in range(num_channels)] , 2) # concat inputs along 3rd dim
        self.y = torch.cat( [self.channel_data[i].y for i in range(num_channels)] , 2) # concat outputs along 3rd dim

        # only need one time vector
        self.tx = self.channel_data[0].tx
        self.ty = self.channel_data[0].ty

class data_grouping():
    def __init__(self, data, input_len=1,output_len=1,tau_offset = 0, inputs_overlap=False, 
                multivariant = False, validation_percent = 10, test_percent = 10, random_split =True, standarize = True):
        
        if multivariant:
            # initiate the input and output tensors
            reshaped_data = multivariate_data(self, data, input_len, output_len, tau_offset, inputs_overlap)
        else:
            reshaped_data = univariate_data(self, data, input_len, output_len, tau_offset, inputs_overlap)

        # # split data by example into specified splitting percentages
        # if validation_percent != 0 or test_percent != 0:
        #     self.train, self.valid, self.test = self.train_valid_test_split(reshaped_data, validation_percent, test_percent)
    
        'Standardize and hold on to mean/var for training, used for inference later'

        'note that if you were going to do some differencing, here would be the place to do it'

        'Next I want to make a data loader and call '


    def train_valid_test_split(self, reshaped_data, validation_percent, test_percent):
        
        n_samples = reshaped_data.shape[0]
        # determine how many samples in each set
        n_train, n_test = int(n_samples * 0.8), int(n_samples * 0.1)
        n_valid = n_samples - n_train - n_test

        # generate random samples to train on


