import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class MLP_SingleTime(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP_SingleTime, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1) * inputs.size(2), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1),inputs.size(2), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class MLPEncoder_SingleTime(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLPEncoder_SingleTime, self).__init__()
        print("==============================",n_in,n_hid)
        self.mlp1 = MLP_SingleTime(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP_SingleTime(n_hid , 2*n_hid, n_hid, do_prob)
        
        self.fc_out = nn.Linear(n_hid, n_hid)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs
        # New shape: [num_sims, num_atoms, num_timesteps,num_dims]
        #pdb.set_trace()
        x = self.mlp1(x)
        x = self.mlp2(x)
        node_embeddings = self.fc_out(x)
        return node_embeddings