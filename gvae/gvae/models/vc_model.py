import torch
from torch import nn

def generate_correlated_data(batch_size, seq_length, agent_num, vel, correlation_factor=0.95):
    """
    generaete correlated data for test
    """
    base_velocity = torch.randn(batch_size, seq_length, vel)
    
    vel_tensor = torch.zeros(batch_size, seq_length, agent_num, vel)
    
    vel_tensor[:, :, 0, :] = base_velocity
    
    for i in range(1, agent_num):
        noise = torch.randn(batch_size, seq_length, vel) * (1 - correlation_factor)
        vel_tensor[:, :, i, :] = base_velocity * correlation_factor + noise
    
    return vel_tensor

class VC_model(nn.Module):
    """
    Linear velocity correlation model
    Return the adjacent matrix to represent attention weights
    """
    def __init__(self):
        super(VC_model, self).__init__()
        self.len = 10

    def forward(self, inputs):
        batch_size, seq_length, agent_num, vel = inputs.shape
        adj_matrix = torch.zeros(batch_size, seq_length-1, agent_num, agent_num, device=inputs.device)
        
        inputs_flat = inputs.view(batch_size, seq_length, agent_num, -1) #[batch_size, agent_num, seq_length * vel]
        inputs_centered = inputs_flat - inputs_flat.mean(dim=1, keepdim=True)
        for s in range(1,seq_length):
            for i in range(agent_num):
                for j in range(i + 1, agent_num):
                    start_id = s-self.len if s-self.len >= 0 else 0
                    vel_i = inputs_centered[:, start_id:s, i].reshape(batch_size, -1)  # [batch_size, input_length * vel]
                    vel_j = inputs_centered[:, start_id:s, j].reshape(batch_size, -1)
                    
                    cov_ij = (vel_i * vel_j).sum(dim=1) / (seq_length * vel - 1)
                    std_i = torch.sqrt((vel_i ** 2).sum(dim=1) / (seq_length * vel - 1))
                    std_j = torch.sqrt((vel_j ** 2).sum(dim=1) / (seq_length * vel - 1))
                    pearson_corr = cov_ij / (std_i * std_j)
                    adj_matrix[:,s-1, i, j] = pearson_corr
                    adj_matrix[:,s-1, j, i] = pearson_corr
        return adj_matrix

if __name__ == '__main__':
    batch_size = 1
    seq_length = 10
    agent_num = 5
    vel = 2
    
    vel_tensor = generate_correlated_data(batch_size,seq_length,agent_num,vel)
    vc_model = VC_model()

    ad = vc_model(vel_tensor)
    print(ad)