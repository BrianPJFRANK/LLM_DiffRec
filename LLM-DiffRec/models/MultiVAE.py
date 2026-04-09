import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiVAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dims must match"
            assert q_dims[-1] == p_dims[0], "Latent dims must match"
            self.q_dims = q_dims

        self.drop = nn.Dropout(dropout)

        # Encoder: [n_items, 600, 200*2]
        self.enc_layer1 = nn.Linear(self.q_dims[0], self.q_dims[1])
        self.enc_layer2 = nn.Linear(self.q_dims[1], self.q_dims[2] * 2)

        # Decoder: [200, 600, n_items]
        self.dec_layer1 = nn.Linear(self.p_dims[0], self.p_dims[1])
        self.dec_layer2 = nn.Linear(self.p_dims[1], self.p_dims[2])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.trunc_normal_(m.bias, std=0.001)

    def forward(self, x):
        # 輸入 L2 正規化
        x = F.normalize(x, p=2, dim=1)
        x = self.drop(x)

        # Encode
        h = torch.tanh(self.enc_layer1(x))
        h = self.enc_layer2(h)
        mu, logvar = torch.chunk(h, 2, dim=1)

        # Reparameterization Trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # Decode
        h = torch.tanh(self.dec_layer1(z))
        logits = self.dec_layer2(h)

        return logits, mu, logvar