import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import ipdb

class COR(nn.Module):
    '''
    item feature is not used in this model, 
    '''
    def __init__(self, mlp_q_dims, mlp_p1_dims, mlp_p2_dims, mlp_p3_dims, \
                                                     item_feature, adj, E1_size, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(COR, self).__init__()
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_dims = mlp_p1_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.mlp_p3_dims = mlp_p3_dims
        self.adj = adj
        self.E1_size = E1_size
        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs
        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid

        # not used in this model, extended for using item feature in future work
        self.item_feature = item_feature
        self.item_learnable_dim = self.mlp_p2_dims[-1]
        self.item_learnable_feat = torch.randn([self.item_feature.size(0), self.item_learnable_dim], \
                                               requires_grad=True).cuda()

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_dims = self.mlp_p1_dims[:-1] + [self.mlp_p1_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims

        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p1_dims[:-1], temp_p1_dims[1:])])
        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)
        
        self.init_weights()

    def reuse_Z2(self,D,E1):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D)
        mu, _ = self.encode(torch.cat((D,E1),1))
        h_p2 = mu
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = F.tanh(h_p2)
            else:
                h_p2 = h_p2[:, :self.mlp_p2_dims[-1]]
        return h_p2
        
    def forward(self, D, E1, Z2_reuse=None, CI=0):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D) # meituan and yelp
        encoder_input = torch.cat((D, E1), 1)  # D,E1
        mu, logvar = self.encode(encoder_input)    # E2 distribution
        E2 = self.reparameterize(mu, logvar)       # E2
        
        if CI == 1: # D=NULL
            encoder_input_0 = torch.cat((torch.zeros_like(D), E1), 1)
            mu_0, logvar_0 = self.encode(encoder_input_0)
            E2_0 = self.reparameterize(mu_0, logvar_0)
            scores = self.decode(E1, E2, E2_0, Z2_reuse)
        else:
            scores = self.decode(E1, E2, None, Z2_reuse)
        reg_loss = self.reg_loss()
        return scores, mu, logvar, reg_loss

    def encode(self, encoder_input):
        h = self.drop(encoder_input)
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h) 
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:, self.mlp_q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, E1, E2, E2_0=None, Z2_reuse=None):

        if E2_0 is None:
            h_p1 = torch.cat((E1, E2), 1)
        else:
            h_p1 = torch.cat((E1, E2_0), 1)
            
        for i, layer in enumerate(self.mlp_p1_layers):
            h_p1 = layer(h_p1)
            if i != len(self.mlp_p1_layers) - 1:
                h_p1 = self.act_function(h_p1)
            else:
                Z1_mu = h_p1[:, :self.mlp_p1_dims[-1]]
                Z1_logvar = h_p1[:, self.mlp_p1_dims[-1]:]

        h_p2 = E2
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)
            else:
                Z2_mu = h_p2[:, :self.mlp_p2_dims[-1]]
                Z2_logvar = h_p2[:, self.mlp_p2_dims[-1]:]

        if Z2_reuse!=None:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
            Z1 = torch.mean(Z1, 0)
            Z2 = Z2_reuse 
        else:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2 = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                    Z2 = torch.unsqueeze(Z2, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2_ = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z2_ = torch.unsqueeze(Z2_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
                    Z2 = torch.cat([Z2,Z2_], 0)
            Z1 = torch.mean(Z1, 0)
            Z2 = torch.mean(Z2, 0)

        user_preference = torch.cat((Z1, Z2), 1)


        h_p3 = user_preference
        for i, layer in enumerate(self.mlp_p3_layers):
            h_p3 = layer(h_p3)
            if i != len(self.mlp_p3_layers) - 1:
                h_p3 = self.act_function(h_p3)
        return h_p3

    def init_weights(self):
        for layer in self.mlp_q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.mlp_p1_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p2_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
        for layer in self.mlp_p3_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        return reg_loss

class COR_G(nn.Module):
    """
    code for adding causal graph
    extension of item feature is not used in this model
    """
    def __init__(self, mlp_q_dims, mlp_p1_1_dims, mlp_p1_2_dims, mlp_p2_dims, mlp_p3_dims, \
                                                         item_feature, adj, E1_size, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(COR_G, self).__init__()
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_1_dims = mlp_p1_1_dims
        self.mlp_p1_2_dims = mlp_p1_2_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.mlp_p3_dims = mlp_p3_dims
        self.adj = adj
        self.E1_size = E1_size
        self.Z1_size = adj.size(0)
        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs

        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid

        # not used in this model, extended for using item feature in future work
        self.item_feature = item_feature
        self.item_learnable_dim = self.mlp_p2_dims[-1]
        self.item_learnable_feat = torch.randn([self.item_feature.size(0), self.item_learnable_dim], \
                                               requires_grad=True).cuda()

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_1_dims = self.mlp_p1_1_dims
        temp_p1_2_dims = self.mlp_p1_2_dims[:-1] + [self.mlp_p1_2_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims

        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p1_1_dims[:-1], temp_p1_1_dims[1:])])
        self.mlp_p1_2_layers = [(torch.randn([self.Z1_size, d_in, d_out],requires_grad=True)).cuda() for
            d_in, d_out in zip(temp_p1_2_dims[:-1], temp_p1_2_dims[1:])]
       
        for i, matrix in enumerate(self.mlp_p1_2_layers):
            temp = torch.unsqueeze(matrix,0) if i==0 else torch.concat((temp,torch.unsqueeze(matrix,0)),0)
        self.mlp_p1_2_layers = nn.Parameter(temp)

        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)
        self.init_weights()

    def reuse_Z2(self,D,E1):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D)
        mu, _ = self.encode(torch.cat((D,E1),1))
        h_p2 = mu
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = F.tanh(h_p2)
            else:
                h_p2 = h_p2[:, :self.mlp_p2_dims[-1]]
        return h_p2

    def forward(self, D, E1, Z2_reuse=None, CI=0):
        if self.bn:
            E1 = self.batchnorm(E1)
        encoder_input = torch.cat((D, E1), 1)  # D,E1
        mu, logvar = self.encode(encoder_input)    # E2 distribution
        E2 = self.reparameterize(mu, logvar)       # E2
        
        if CI == 1: # D=NULL
            encoder_input_0 = torch.cat((torch.zeros_like(D), E1), 1)
            mu_0, logvar_0 = self.encode(encoder_input_0)
            E2_0 = self.reparameterize(mu_0, logvar_0)
            scores = self.decode(E1, E2, E2_0, Z2_reuse)
        else:
            scores = self.decode(E1, E2, None, Z2_reuse)
        reg_loss = self.reg_loss()
        return scores, mu, logvar, reg_loss
    
    def encode(self, encoder_input):
        h = self.drop(encoder_input)
        
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h)
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:, self.mlp_q_dims[-1]:]
                
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, E1, E2, E2_0=None, Z2_reuse=None):

        if E2_0 is None:
            h_p1 = torch.cat((E1, E2), 1)
        else:
            h_p1 = torch.cat((E1, E2_0), 1)
        h_p1 = torch.unsqueeze(h_p1, -1)
        for i, layer in enumerate(self.mlp_p1_1_layers):
            h_p1 = layer(h_p1)
            if i != len(self.mlp_p1_1_layers) - 1:
                h_p1 = self.act_function(h_p1)
        h_p1 = torch.matmul(self.adj, h_p1)
        h_p1 = torch.unsqueeze(h_p1, 2)
        for i, matrix in enumerate(self.mlp_p1_2_layers):
            h_p1 = torch.matmul(h_p1, matrix)
            if i != len(self.mlp_p1_2_layers) - 1:
                h_p1 = self.act_function(h_p1)
            else:
                h_p1 = torch.squeeze(h_p1)
                Z1_mu = torch.squeeze(h_p1[:, :, :self.mlp_p1_2_dims[-1]])
                Z1_logvar = torch.squeeze(h_p1[:, :, self.mlp_p1_2_dims[-1]:])
        
        h_p2 = E2
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)
            else:
                Z2_mu = h_p2[:, :self.mlp_p2_dims[-1]]
                Z2_logvar = h_p2[:, self.mlp_p2_dims[-1]:]

        if Z2_reuse!=None:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
            Z1 = torch.mean(Z1, 0)
            Z2 = Z2_reuse 
        else:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2 = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                    Z2 = torch.unsqueeze(Z2, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2_ = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z2_ = torch.unsqueeze(Z2_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
                    Z2 = torch.cat([Z2,Z2_], 0)
            Z1 = torch.mean(Z1, 0)
            Z2 = torch.mean(Z2, 0)

        user_preference = torch.cat((Z1, Z2), 1)

        h_p3 = user_preference
        for i, layer in enumerate(self.mlp_p3_layers):
            h_p3 = layer(h_p3)
            if i != len(self.mlp_p3_layers) - 1:
                h_p3 = self.act_function(h_p3)
        return h_p3
    
    def init_weights(self):
        for layer in self.mlp_q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.mlp_p1_1_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
        for matrix in self.mlp_p1_2_layers:
            # Xavier Initialization for weights
            size = matrix.data.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            matrix.data.normal_(0.0, std)
            
        for layer in self.mlp_p2_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
        for layer in self.mlp_p3_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p1_1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        return reg_loss

def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD