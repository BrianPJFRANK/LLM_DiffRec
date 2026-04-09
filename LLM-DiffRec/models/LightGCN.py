import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, adj_matrix_cpu):

        all_emb_npu = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [all_emb_npu]

        all_emb_cpu = all_emb_npu.cpu()

        for layer in range(self.n_layers):
            all_emb_cpu = torch.sparse.mm(adj_matrix_cpu, all_emb_cpu)
            embs.append(all_emb_cpu.to(self.user_emb.weight.device))

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def bpr_loss(self, users, pos_items, neg_items, all_users, all_items):
        u_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        # Scoring the inner product
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)

        # BPR Loss
        loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))

        # L2 Regularization
        reg_loss = (1/2) * (self.user_emb.weight[users].norm(2).pow(2) +
                            self.item_emb.weight[pos_items].norm(2).pow(2) +
                            self.item_emb.weight[neg_items].norm(2).pow(2)) / float(len(users))
        return loss, reg_loss