import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import datetime
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.MultiVAE import MultiVAE
from utils import evaluate_utils, data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='amazon-instruments', help='dataset name')
parser.add_argument('--data_path', type=str, default='../datasets/', help='data path')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--cuda', action='store_true', help='use CUDA/NPU')
parser.add_argument('--gpu', type=str, default='0', help='gpu/npu card ID')
parser.add_argument('--total_anneal_steps', type=int, default=200000, help='KL annealing steps')
parser.add_argument('--anneal_cap', type=float, default=0.2, help='KL annealing cap')

args = parser.parse_args()

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

log_dir = f"./log/{args.dataset}_baselines"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"multivae_bs{args.batch_size}_{current_time}.txt")
sys.stdout = Logger(log_filename)

print(f"Logging output to: {log_filename}")


if args.cuda and hasattr(torch, 'npu') and torch.npu.is_available():
    import torch_npu
    device = torch.device(f"npu:{args.gpu}")
    print(f"Using NPU Device: npu:{args.gpu}")
elif args.cuda and torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

train_path = os.path.join(args.data_path, args.dataset, 'train_list.npy')
valid_path = os.path.join(args.data_path, args.dataset, 'valid_list.npy')
test_path = os.path.join(args.data_path, args.dataset, 'test_list.npy')

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
mask_tv = train_data + valid_y_data

train_tensor = torch.FloatTensor(train_data.toarray())
train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

p_dims = [200, 600, n_item]
model = MultiVAE(p_dims).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

def loss_function(recon_x, x, mu, logvar, anneal):
    # Multinomial Log Likelihood
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # KL Divergence
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD

def evaluate_vae(model, test_data_matrix, mask_matrix, topN):
    model.eval()
    predict_items, target_items = [], []
    e_N = mask_matrix.shape[0]

    for i in range(e_N):
        target_items.append(test_data_matrix[i, :].nonzero()[1].tolist())

    with torch.no_grad():
        test_tensor = torch.FloatTensor(train_data.toarray()).to(device)
        for batch_idx in range(0, e_N, args.batch_size):
            end_idx = min(batch_idx + args.batch_size, e_N)
            batch = test_tensor[batch_idx:end_idx]

            logits, _, _ = model(batch)
            logits[mask_matrix[batch_idx:end_idx].toarray() > 0] = -np.inf

            _, indices = torch.topk(logits, topN[-1])
            predict_items.extend(indices.cpu().numpy().tolist())

    return evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

update_count = 0
best_recall, best_epoch = -100, 0
best_valid_res, best_test_res = None, None

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch[0].to(device)

        anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
        update_count += 1

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar, anneal)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    if epoch % 5 == 0:
        valid_res = evaluate_vae(model, valid_y_data, train_data, eval(args.topN))
        test_res = evaluate_vae(model, test_y_data, mask_tv, eval(args.topN))
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f}")
        evaluate_utils.print_results(None, valid_res, test_res)

        if valid_res[1][1] > best_recall:
            best_recall, best_epoch = valid_res[1][1], epoch
            best_valid_res, best_test_res = valid_res, test_res

print(f"Best Epoch: {best_epoch:03d}")
evaluate_utils.print_results(None, best_valid_res, best_test_res)