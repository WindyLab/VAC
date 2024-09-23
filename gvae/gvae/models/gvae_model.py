import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from . import model_utils
from .encoders import *
from .model_utils import RefNRIMLP, encode_onehot
import math
import pdb
from torch import Tensor

def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class GVAE(nn.Module):
    def __init__(self, params):
        super(GVAE, self).__init__()
        # Model Params
        self.num_vars = params['num_vars']
        self.encoder = Encoder(params)
        decoder_type = params.get('decoder_type', None)
        if decoder_type == 'ref_mlp':
            self.decoder = DNRI_MLP_Decoder(params)
        else:
            self.decoder = DNRI_Decoder(params)
        self.num_edge_types = params.get('num_edge_types')

        # Training params
        self.gumbel_temp = params.get('gumbel_temp')
        self.train_hard_sample = params.get('train_hard_sample')
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)
        
        self.normalize_kl = params.get('normalize_kl', False)
        self.normalize_kl_per_var = params.get('normalize_kl_per_var', False)
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.kl_coef = params.get('kl_coef', 1.)

        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.timesteps = params.get('timesteps', 0)
        self.burn_in_steps = params.get('train_burn_in_steps')
        self.teacher_forcing_prior = params.get('teacher_forcing_prior', False)
        self.val_teacher_forcing_steps = params.get('val_teacher_forcing_steps', -1)
        self.add_uniform_prior = params.get('add_uniform_prior')
        if self.add_uniform_prior:
            if params.get('no_edge_prior') is not None:
                prior = np.zeros(self.num_edge_types)
                prior.fill((1 - params['no_edge_prior'])/(self.num_edge_types - 1))
                prior[0] = params['no_edge_prior']
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior
                print("USING NO EDGE PRIOR: ",self.log_prior)
            else:
                print("USING UNIFORM PRIOR")
                prior = np.zeros(self.num_edge_types)
                prior.fill(1.0/self.num_edge_types)
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior

    def single_step_forward(self,inputs, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types),
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        predictions, decoder_hidden = self.decoder(inputs, decoder_hidden, edges)
        return predictions, decoder_hidden, edges

    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False):
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_predictions = []
        hard_sample = (not is_train) or self.train_hard_sample
        prior_logits, posterior_logits, _,node_embeddings = self.encoder(inputs[:, :-1])
        if not is_train:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        for step in range(num_time_steps-1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions
            if not use_prior_logits:
                current_p_logits = posterior_logits[:, step]
            else:
                current_p_logits = prior_logits[:, step]

            # print(f"current_inputs shape{current_inputs.shape}, \
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_p_logits, hard_sample)
            #      decoder_hidden shape{decoder_hidden.shape}, \
            #      edges shape {edges.shape}")
            all_predictions.append(predictions)

        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        loss_nll = self.nll(all_predictions, target)
        prob = F.softmax(posterior_logits, dim=-1)
        loss_kl = self.kl_categorical_learned(prob, prior_logits)
        if self.add_uniform_prior:
            loss_kl = 0.5*loss_kl + 0.5*self.kl_categorical_avg(prob)
        loss = loss_nll + self.kl_coef*loss_kl
        loss = loss.mean()
        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, posterior_logits, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        burn_in_timesteps = inputs.size(1)
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        all_predictions = []
        all_edges = []
        prior_logits, _, prior_hidden = self.encoder(inputs[:, :-1])
        for step in range(burn_in_timesteps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
            if return_everything:
                all_edges.append(edges)
                all_predictions.append(predictions)
        predictions = inputs[:, burn_in_timesteps-1]
        for step in range(prediction_steps):
            current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
            predictions, decoder_hidden, edges = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
            all_predictions.append(predictions)
            all_edges.append(edges)
        
        predictions = torch.stack(all_predictions, dim=1)
        if return_edges:
            edges = torch.stack(all_edges, dim=1)
            return predictions, edges
        else:
            return predictions

    def copy_states(self, state):
        if isinstance(state, tuple) or isinstance(state, list):
            current_state = (state[0].clone(), state[1].clone())
        else:
            current_state = state.clone()
        return current_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            return (result0, result1)
        else:
            return torch.cat(hidden, dim=0)

        print("INPUT SHAPE: ",inputs.shape)
    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size, return_edges=False):
        prior_logits, _, prior_hidden = self.encoder(inputs[:, :-1])
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        for step in range(burn_in_steps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, _ = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
        all_timestep_preds = []
        all_timestep_edges = []
        for window_ind in range(burn_in_steps - 1, inputs.size(1)-1, batch_size):
            current_batch_preds = []
            current_batch_edges = []
            prior_states = []
            decoder_states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                predictions = inputs[:, window_ind + step] 
                current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
                predictions, decoder_hidden, _ = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
                current_batch_preds.append(predictions)
                tmp_prior = self.encoder.copy_states(prior_hidden)
                tmp_decoder = self.copy_states(decoder_hidden)
                prior_states.append(tmp_prior)
                decoder_states.append(tmp_decoder)
                if return_edges:
                    current_batch_edges.append(current_edge_logits.cpu())
            batch_prior_hidden = self.encoder.merge_hidden(prior_states)
            batch_decoder_hidden = self.merge_hidden(decoder_states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            if return_edges:
                current_batch_edges = torch.cat(current_batch_edges, 0)
                current_timestep_edges = [current_batch_edges]
            for step in range(prediction_steps - 1):
                current_batch_edge_logits, batch_prior_hidden = self.encoder.single_step_forward(current_batch_preds, batch_prior_hidden)
                current_batch_preds, batch_decoder_hidden, _ = self.single_step_forward(current_batch_preds, batch_decoder_hidden, current_batch_edge_logits, True)
                current_timestep_preds.append(current_batch_preds)
                if return_edges:
                    current_timestep_edges.append(current_batch_edge_logits.cpu())
            all_timestep_preds.append(torch.stack(current_timestep_preds, dim=1))
            if return_edges:
                all_timestep_edges.append(torch.stack(current_timestep_edges, dim=1))
        result =  torch.cat(all_timestep_preds, dim=0)
        if return_edges:
            edge_result = torch.cat(all_timestep_edges, dim=0)
            return result.unsqueeze(0), edge_result.unsqueeze(0)
        else:
            return result.unsqueeze(0)

    def nll(self, preds, target):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target)

    def nll_gaussian(self, preds, target, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        #neg_log_p += const
        if self.normalize_nll_per_var:
            return neg_log_p.sum() / (target.size(0) * target.size(2))
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const).view(preds.size(0), -1).mean(dim=1)
        else:
            return neg_log_p.view(target.size(0), -1).sum() / (target.size(1))


    def nll_crossent(self, preds, target):
        if self.normalize_nll:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def nll_poisson(self, preds, target):
        if self.normalize_nll:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds*(torch.log(preds + 1e-16) - log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_avg(self, preds, eps=1e-16):
        avg_preds = preds.mean(dim=2)
        kl_div = avg_preds*(torch.log(avg_preds+eps) - self.log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

from torch_geometric.nn import MessagePassing
from torch import Tensor

class EdgeDecoder(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2*in_channels, 2*out_channels),
            torch.nn.ELU(),
            nn.Dropout(p=0.4),
            torch.nn.Linear(2*out_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor,edge_weight) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        #pdb.set_trace()
        return self.propagate(edge_index, x=x,edge_weight=edge_weight)  # shape [num_nodes, out_channels]

    def message(self, x_j: Tensor, x_i: Tensor,edge_weight) -> Tensor:
        # x_j: Source node features of shape [num_edges, in_channels]
        # x_i: Target node features of shape [num_edges, in_channels]
        edge_features = torch.cat( [x_i,x_j], dim=-1)
        #print(f'xi {x_i.shape} xj {x_j.shape} {edge_features.shape} {edge_weight.shape}')
        #pdb.set_trace()
        edge_weight = edge_weight.unsqueeze(0).unsqueeze(-1)
        return self.mlp( edge_features * edge_weight )  # shape [num_edges, out_channels]
    
class Encoder(nn.Module):
    # Here, encoder also produces prior
    def __init__(self, params):
        super(Encoder, self).__init__()
        num_vars = params['num_vars']
        self.num_edges = params['num_edge_types']
        self.sepaate_prior_encoder = params.get('separate_prior_encoder', False)
        no_bn = False
        dropout = params['encoder_dropout']
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
        self.save_eval_memory = params.get('encoder_save_eval_memory', False)


        hidden_size = params['encoder_hidden']
        self.encoder_hidden = hidden_size
        rnn_hidden_size = params['encoder_rnn_hidden']
        rnn_type = params['encoder_rnn_type']
        inp_size = params['input_size']
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)
        
        ###################### layers #####################
        self.node_embedding_encoders = MLPEncoder_SingleTime(inp_size,
                                      hidden_size,
                                      1,
                                      dropout).cuda()
        from torch_geometric.nn import EdgeConv
        self.edge_conv = EdgeConv(nn=torch.nn.Linear(hidden_size*2, hidden_size), aggr='max')
        self.edge_conv_decoder = EdgeConv(nn=torch.nn.Linear(hidden_size*2, hidden_size), aggr='max')
        
        self.edge_encoders = nn.Sequential(nn.Linear(2*self.encoder_hidden, 2*self.encoder_hidden),
                                           nn.ELU(),
                                           nn.Dropout(p=0.2),
                                           nn.Linear(2*self.encoder_hidden, self.encoder_hidden)).cuda()
        self.edge_self_encoder = nn.Sequential(nn.Linear(1, 2*self.encoder_hidden),
                                           nn.ELU(),
                                           nn.Dropout(p=0.2),
                                           nn.Linear(2*self.encoder_hidden, self.encoder_hidden)).cuda()
        ###################### ##################### #####################

        if rnn_hidden_size is None:
            rnn_hidden_size = hidden_size
        if rnn_type == 'lstm':
            self.forward_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
        out_hidden_size = 2*rnn_hidden_size
        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.encoder_fc_out = nn.Linear(out_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            layers = [nn.Linear(out_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.encoder_fc_out = nn.Sequential(*layers)
        #pdb.set_trace()
        num_layers = params['prior_num_layers']
        if num_layers == 1:
            self.prior_fc_out = nn.Linear(rnn_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['prior_hidden_size']
            layers = [nn.Linear(rnn_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.prior_fc_out = nn.Sequential(*layers)


        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge_index = torch.tensor([[i, j] for i in range(num_vars) for j in range(num_vars) if i != j], dtype=torch.long).t().cuda()
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def copy_states(self, prior_state):
        if isinstance(prior_state, tuple) or isinstance(prior_state, list):
            current_prior_state = (prior_state[0].clone(), prior_state[1].clone())
        else:
            current_prior_state = prior_state.clone()
        return current_prior_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            result = (result0, result1)
        else:
            result = torch.cat(hidden, dim=0)
        return result

    def edge_attention(self,x,edge_index):
        """
        compute edge features from node embeddings
        return [batch_size * time * edge_num] edge logits.
        """
        batch_sz = x.shape[0]
        N = x.shape[1]  ## x :[batch,N,time,feat_dim]
        total_time = x.shape[2]
        all_edge_features = torch.zeros((batch_sz,total_time,edge_index.shape[1],self.encoder_hidden)).cuda()
       # print(f'edge_index shape {edge_index.shape}')
        #pdb.set_trace()
        for t in range(total_time):
            for e_id in range(edge_index.shape[1]):
                i = edge_index[0][e_id]
                j = edge_index[1][e_id]
                edge_fe = torch.cat((x[:,i,t],x[:,j,t]),dim=1)
                e = self.edge_encoders(edge_fe)
                all_edge_features[:,t,e_id,:] = e
        return all_edge_features

    def edge_self_attention(self,x,edge_index):
        """
        compute edge features from node embeddings
        return [batch_size * time * edge_num] edge logits.
        """
        batch_sz = x.shape[0]
        N = x.shape[1]  ## x :[batch,N,time,feat_dim]
        total_time = x.shape[2]
        all_edge_features = torch.zeros((batch_sz,total_time,edge_index.shape[1],self.encoder_hidden)).cuda()
        # print(f'edge_index shape {edge_index.shape}')
        #  pdb.set_trace()
        for t in range(total_time):
            for e_id in range(edge_index.shape[1]):
                i = edge_index[0][e_id]
                j = edge_index[1][e_id]

                x_i = x[:, i, t].unsqueeze(2)  # [batch_size, feature_dim, 1]
                x_j = x[:, j, t].unsqueeze(1)  # [batch_size, 1, feature_dim]

                e = torch.bmm(x_j,x_i) / x_j.shape[-1]
                e = self.edge_self_encoder(e.squeeze(2))
                all_edge_features[:,t,e_id,:] = e
        return all_edge_features

    def forward(self, inputs):
        # Inputs is shape [batch, num_timesteps, num_vars, input_size]
        x = inputs.transpose(2, 1).contiguous()
        # New shape: [num_sims, num_atoms, num_timesteps, num_dims]
        ##############################################
        node_embeddings = self.node_embedding_encoders(x.contiguous())
      #  print(f'node_embeddings shape {node_embeddings.shape}')

        node_embeddings = self.edge_conv(node_embeddings,self.edge_index)
     #   print(f'node_embeddings shape {node_embeddings.shape}')
        
       # all_edge_features = self.edge_attention(node_embeddings,self.edge_index)
        all_edge_features = self.edge_self_attention(node_embeddings,self.edge_index)
        
        x = all_edge_features.permute(0,2,1,3)

        ##############################################
        
        # At this point, x should be [batch, num_edges, num_timesteps, hidden_size]
        # RNN aggregation
        old_shape = x.shape
        x = x.contiguous().view(-1, old_shape[2], old_shape[3])
        forward_x, prior_state = self.forward_rnn(x)
        timesteps = old_shape[2]
        reverse_x = x.flip(1)
        reverse_x, _ = self.reverse_rnn(reverse_x)
        reverse_x = reverse_x.flip(1)
        
        #x: [batch*num_edges, num_timesteps, hidden_size]
        prior_result = self.prior_fc_out(forward_x).view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1,2).contiguous()
        combined_x = torch.cat([forward_x, reverse_x], dim=-1)
        encoder_result = self.encoder_fc_out(combined_x).view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1,2).contiguous()
        return prior_result, encoder_result, prior_state,node_embeddings

class DNRI_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_Decoder, self).__init__()
        self.num_vars = num_vars =  params['num_vars']
        input_size = params['input_size']
        self.gpu = params['gpu']
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)
        
        ##############
        self.edge_index = torch.tensor([[i, j] for i in range(num_vars) for j in range(num_vars) if i != j], dtype=torch.long).t().cuda()
        self.edge_decoder = EdgeDecoder(n_hid,n_hid).cuda()
        ##############

    def get_initial_hidden(self, inputs):
        return torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def convert_edge_index(self,edges_weight):
        w,id = torch.max(edges_weight,dim=-1)
        return w

    def forward(self, inputs, hidden, edges):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        # Edges size: [batch, num_edges, num_edge_types]
        if self.training:
            dropout_prob = self.dropout_prob
        else:
            dropout_prob = 0.
        w_ = self.convert_edge_index(edges)
        agg_msgs2 = self.edge_decoder(hidden,self.edge_index,w_ ).squeeze(0)

        ############################################################
        # GRU-style gated aggregation
        inp_r = self.input_r(inputs)
        inp_i = self.input_i(inputs)
        inp_n = self.input_n(inputs)
        
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs2))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs2))
        n = torch.tanh(inp_n + r*self.hidden_h(agg_msgs2))
        hidden = (1 - i)*n + i*hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=dropout_prob)
        pred = self.out_fc3(pred)

        pred = inputs + pred

        return pred, hidden