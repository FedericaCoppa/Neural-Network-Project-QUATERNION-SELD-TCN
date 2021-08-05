from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter

#from core.quaternion_ops import *
from .quaternion_ops import *
from keras.layers import Layer




class QuaternionBatchNorm(Layer):
    

    def __init__(self, num_features, gamma_init=1., beta_param=True, training=True):
        super(QuaternionBatchNorm, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.training = training
        self.eps = torch.tensor(1e-5)

    def reset_parameters(self):
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):

        quat_components = torch.chunk(input, 4, dim=1)

        h_r, h_i, h_j, h_k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
        
        mean_r= torch.mean(h_r)
        mean_i=torch.mean(h_i)
        mean_j=torch.mean(h_j)
        mean_k=torch.mean(h_k)

        
        quat_variance = torch.mean(((h_r - mean_r)**2 + (h_i - mean_i)**2 + (h_j - mean_j)**2 + (h_k - mean_k)**2))

        denominator = torch.sqrt(quat_variance + self.eps)

        h_r_norm=h_r/denominator
        h_i_norm=h_i/denominator
        h_j_norm=h_j/denominator
        h_k_norm=h_k/denominator

        mean_norm_r= mean_r/denominator
        mean_norm_i=mean_i/denominator
        mean_norm_j=mean_j/denominator
        mean_norm_k=mean_k/denominator


        beta_components = torch.chunk(self.beta, 4, dim=1)

        comp1=(self.gamma *(h_r_norm - mean_norm_r))+beta_components[0]
        comp2=(self.gamma*(h_i_norm - mean_norm_i))+beta_components[1]
        comp3=(self.gamma*(h_j_norm - mean_norm_j))+beta_components[2]
        comp4=(self.gamma*(h_k_norm-mean_norm_k))+beta_components[3]

        y_output = torch.cat((comp1, comp2, comp3, comp4), dim=1)

        return y_output

         
        
        
        
        

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma) \
               + ', beta=' + str(self.beta) \
               + ', eps=' + str(self.eps) + ')'
