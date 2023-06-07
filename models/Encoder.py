import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Decoder import dict2obj

__all__ = (
    'Encoder_HighWay',
)

class HighWay(nn.Module):
    def __init__(self, hidden_size, with_gate=True):
        super(HighWay, self).__init__()
        self.with_gate = with_gate
        self.w1 = nn.Linear(hidden_size, hidden_size)

        if self.with_gate:
            self.w2 = nn.Linear(hidden_size, hidden_size)
            
        self.tanh = nn.Tanh()
        #self._init_weights()

    def forward(self, x):
        y = self.tanh(self.w1(x))
        if self.with_gate:
            gate = torch.sigmoid(self.w2(x))
            return gate * x + (1 - gate) * y
        else:
            return x + y

class ORG(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ORG, self).__init__()
        '''
        Object Relational Graph (ORG) is a module that learns 
        to describe an object based on its relationship 
        with others in a video.
        
        Arguments:
            feat_size : The object feature size that obtained from
                        the last fully-connected layer of the backbone
                        of Faster R-CNN
        '''
        
        self.dropout = nn.Dropout(0.3)# dropout rate 0.3
        self.adjacency_dropout = nn.Dropout(0.5)# dropout rate 0.5

        self.object_projection = nn.Linear(in_features=input_dim,
                                           out_features=output_dim)
        self.sigma_r = nn.Linear(in_features=input_dim,
                                 out_features=output_dim)
        self.psi_r = nn.Linear(in_features=input_dim,
                               out_features=output_dim)
        self.w_r = nn.Linear(in_features=input_dim, 
                             out_features=output_dim, 
                             bias=False)
    
    def forward(self, object_variable): # only one inputs
        '''
        input:
          r_feats: The embedded feature into 512-D which has shape of
                   (batch_size, feature_dim, num_frames, num_objects)
        output:
          r_hat: Enhanced features that has the information of relation
                 between objects 
                 (batch_size, num_frames, num_objects, feature_dim)
        '''
        object_variable = self.dropout(object_variable)  ## Projected Object Features to 512-D
        r_feat = self.object_projection(object_variable)  
        sigma_r = self.sigma_r(object_variable)   ## Sigma(R) = R . Wi + bi
        psi_r = self.psi_r(object_variable)   ## Psi(R) = R . Wj + bj

        a_coeff = torch.bmm(sigma_r.view(-1, r_feat.size(-2), r_feat.size(-1)),  ## A = Simga(R) . Psi(R).T
                            psi_r.contiguous().view(-1, r_feat.size(-2), r_feat.size(-1))\
                            .transpose(1, 2)).view(r_feat.size(0), r_feat.size(1), r_feat.size(-2), r_feat.size(-2))
        a_hat = F.softmax(a_coeff, dim=-1)  ## A_hat = Softmax(A)

        a_hat = self.adjacency_dropout(a_hat)  ## Applying Dropout        
        r_hat = torch.matmul(a_hat, self.w_r(object_variable))  ## R_hat = A_hat . R . Wr
        
        return r_feat, r_hat

class MultipleStreams(nn.Module):
    def __init__(self, opt, module_func, org_func, is_rnn=False):
        super(MultipleStreams, self).__init__()
        self.encoders = []

        modality = opt['modality'].lower() # this result in 'mi' and 'mio' in the future
        for char in modality: # iterate every character in modality
            input_dim = opt.get('dim_' + char, None) 
            output_dim = opt.get('dim_hidden', 512)
            dropout = opt.get('encoder_dropout', 0.5)
            assert input_dim is not None, \
                'The modality is {}, but dim_{} can not be found in opt'.format(modality, char)
            
            if char != 'o': # jika modul bukan untuk objek fitur
                module = module_func(input_dim, output_dim, dropout)
            else:
                module = org_func(input_dim, output_dim) 
            
            self.add_module("Encoder_%s" % char.upper(), module)
            self.encoders.append(module)
 
        self.num_feats = len(modality)
        self.is_rnn = is_rnn

    def forward(self, input_feats):
        assert self.num_feats == len(input_feats)
        if not self.is_rnn:
            encoder_ouputs = [encoder(feats) for encoder, feats in zip(self.encoders, input_feats)]
            encoder_hiddens = [item.mean(1) for i, item in enumerate(encoder_ouputs) if i < 2]
        else:
            pass
            # TODO
        
        if getattr(self, 'subsequent_processing', None) is not None:
            return self.subsequent_processing(encoder_ouputs, encoder_hiddens)

        return encoder_ouputs, encoder_hiddens


class Encoder_HighWay(MultipleStreams):
    def __init__(self, opt):
        with_gate = opt.get('gate', True)
        module_func = lambda x, y, z: nn.Sequential(nn.Linear(x, y), HighWay(y, with_gate), nn.Dropout(z))
        org_func = lambda input_shape, output_shape: ORG(input_shape, output_shape) # ini nanti dibawa ke yang multiple stream
        super(Encoder_HighWay, self).__init__(opt, module_func, org_func)
