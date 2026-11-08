import torch
import torch.nn as nn
import torch.nn.functional as F


class MRFPropagation(nn.Module):
    def __init__(self, num_nodes, node_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_features, node_features),
            nn.GELU(),
            nn.LayerNorm(node_features),
        )
        self.node_potential_mlp = nn.Sequential(
            nn.Linear(node_features, 1),
            nn.GELU(),
            nn.Tanh()  # 或者加入 torch.clamp 限制范围
        )
        self.edge_potential_mlp = nn.Sequential(
            nn.Linear(2 * node_features, 1),
            nn.Softplus()
        )
        # self.edge_index = torch.tensor([[0, 1], [0, 2], [3, 4], [2, 4], [4, 6], [5, 9]], dtype=torch.long)
        self.edge_index = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 4], 3: [1, 5], 4: [2, 6], 5: [3, 6, 7, 11],
                          6: [4, 5, 8, 12],
                          7: [5, 9], 8: [6, 10], 9: [7], 10: [8], 11: [5, 12, 13], 12: [6, 11, 14], 13: [11, 15],
                          14: [12, 16],
                          15: [13], 16: [14]}


    def update(self, x, messages):
        # Update with residual connection
        # return x + self.mlp((x + messages).unsqueeze(1)).squeeze(1)
        return  self.mlp((x + messages).unsqueeze(1)).squeeze(1)

    def edge_potentials(self, x_i, x_j):
        # Calculate edge potentials between node i and node j
        return self.edge_potential_mlp(torch.cat([x_i, x_j], dim=-1))

    def message_update(self, x):
        b,n,c = x.size()
        messages = torch.zeros(b, n, 1, device=x.device)


        # 对于每条边，分别计算消息
        for node_idx , neighbor_node_list in self.edge_index.items():

            node = x[:, node_idx, :].unsqueeze(1).repeat(1, len(neighbor_node_list), 1)
            edge = x[:, neighbor_node_list, :]
            potential = self.edge_potentials(node,edge )
            potential = torch.log1p(torch.exp(potential))  # log-sum-exp技巧
            messages[:, node_idx, :] = torch.sum(potential.reshape(b,-1) , dim = -1, keepdim=True)



        node_potentials = self.node_potential_mlp(x)
        # messages = torch.sum(messages, dim=-1, keepdim=True) * node_potentials
        messages = messages + node_potentials


        return messages

    # def message_update(self, x):
    #     b, n, c = x.size()
    #     messages = torch.zeros(b, n, 1, device=x.device)
    #
    #     # # 准备节点特征的矩阵形式
    #     x_matrix = x.unsqueeze(1)  # (b, 1,n c)
    #     x_matrix = x_matrix.expand(-1, n, n, -1)  # (b, n, n, c)
    #     # x_matrix = x
    #
    #     # 准备边势的计算
    #     edge_potentials = torch.zeros(b, n, 1, device=x.device)
    #
    #     # 计算所有可能的边势
    #     for node_idx, neighbor_node_list in self.edge_index.items():
    #         neighbor_indices = torch.tensor(neighbor_node_list, dtype=torch.long, device=x.device)
    #         # 提取节点和邻居节点的特征
    #         node_features = x_matrix[:, :len(neighbor_node_list),node_idx,  :].reshape(b, len(neighbor_node_list), c)
    #         neighbor_features = x[:,  neighbor_indices, :]
    #         # 计算边势
    #         potentials = self.edge_potential_mlp(torch.cat([node_features, neighbor_features], dim=-1))
    #         edge_potentials[:, node_idx, :] += potentials.sum(dim=1)
    #
    #     # 应用log-sum-exp技巧
    #     edge_potentials = torch.log1p(torch.exp(edge_potentials))
    #
    #     # 计算节点势
    #     node_potentials = self.node_potential_mlp(x)  # (b, n, 1)
    #
    #     # 合并边势和节点势
    #     messages = edge_potentials + node_potentials
    #
    #     return messages


    def forward(self, x):


        messages = self.message_update(x)
        node_states = torch.zeros_like(x)


        for i in range(x.size(1)):
            neighbors_messages = messages[:, i, :]
            node_states[:, i, :] = self.update(x[:, i, :], neighbors_messages)


        return node_states


if __name__ == '__main__':
    import time
    # Test the model
    x = torch.randn(16, 17, 1024).to('cuda')

    mrf_propagation = MRFPropagation(num_nodes=17, node_features=1024).to('cuda')
    time1 = time.time()
    out = mrf_propagation(x)
    print(time.time() - time1)
    print(out.shape)
    total_params = sum(p.numel() for p in mrf_propagation.parameters())
    print("=>参数量为:{}".format(total_params))
