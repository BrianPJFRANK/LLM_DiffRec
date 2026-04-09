import argparse
import os
import scipy.sparse as sp
import numpy as np
import torch
import torch.optim as optim
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.LightGCN import LightGCN
from utils import evaluate_utils, data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='amazon-instruments')
parser.add_argument('--data_path', type=str, default='../datasets/')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=str, default='0')

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

save_dir = f"./saved_models_baselines"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"lightgcn_L{args.layers}_bs{args.batch_size}_{current_time}.txt")
sys.stdout = Logger(log_filename)

print(f"✅ Logging output to: {log_filename}")

# NPU/GPU 適配邏輯
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

# 動態構建 User-Item 歸一化二分圖鄰接矩陣 (Normalized Adjacency Matrix)
def build_adj_mat(R, n_users, n_items):
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()

    # D^-1/2 * A * D^-1/2
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv).tocsr()

    coo = norm_adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=torch.device('cpu'))

print("Building adjacency matrix...")
norm_adj_tensor = build_adj_mat(train_data, n_user, n_item)

model = LightGCN(n_user, n_item, emb_dim=64, n_layers=args.layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 負採樣邏輯
def bpr_sampling(train_data, batch_size):
    users, pos_items, neg_items = [], [], []
    train_coo = train_data.tocoo()
    user_item_dict = {u: set() for u in range(n_user)}
    for u, i in zip(train_coo.row, train_coo.col):
        user_item_dict[u].add(i)

    for _ in range(batch_size):
        u = np.random.randint(0, n_user)
        while len(user_item_dict[u]) == 0:
            u = np.random.randint(0, n_user)
        i = np.random.choice(list(user_item_dict[u]))
        j = np.random.randint(0, n_item)
        while j in user_item_dict[u]:
            j = np.random.randint(0, n_item)

        users.append(u)
        pos_items.append(i)
        neg_items.append(j)
    return torch.tensor(users), torch.tensor(pos_items), torch.tensor(neg_items)

# 評估函數
def evaluate_lightgcn(model, test_data_matrix, mask_matrix, topN):
    model.eval()
    predict_items, target_items = [], []
    e_N = mask_matrix.shape[0]

    for i in range(e_N):
        target_items.append(test_data_matrix[i, :].nonzero()[1].tolist())

    with torch.no_grad():
        all_users, all_items = model(norm_adj_tensor)
        for batch_idx in range(0, e_N, args.batch_size):
            end_idx = min(batch_idx + args.batch_size, e_N)
            u_batch = all_users[batch_idx:end_idx]

            # 矩陣內積打分
            logits = torch.matmul(u_batch, all_items.T)
            logits[mask_matrix[batch_idx:end_idx].toarray() > 0] = -np.inf

            _, indices = torch.topk(logits, topN[-1])
            predict_items.extend(indices.cpu().numpy().tolist())

    return evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

best_recall, best_epoch = -100, 0
best_valid_res, best_test_res = None, None

for epoch in range(1, args.epochs + 1):
    model.train()
    users, pos_items, neg_items = bpr_sampling(train_data, args.batch_size * 50) # 每個 epoch 採樣一定數量
    users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)

    optimizer.zero_grad()
    all_users, all_items = model(norm_adj_tensor)
    bpr_loss, reg_loss = model.bpr_loss(users, pos_items, neg_items, all_users, all_items)
    loss = bpr_loss + args.wd * reg_loss
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        valid_res = evaluate_lightgcn(model, valid_y_data, train_data, eval(args.topN))
        test_res = evaluate_lightgcn(model, test_y_data, mask_tv, eval(args.topN))
        print(f"Epoch {epoch:03d} | BPR Loss: {bpr_loss.item():.4f}")
        evaluate_utils.print_results(None, valid_res, test_res)

        if valid_res[1][1] > best_recall:
            best_recall, best_epoch = valid_res[1][1], epoch
            best_valid_res, best_test_res = valid_res, test_res
            save_model_path = os.path.join(save_dir, f'lightgcn_L{args.layers}_best.pth')
            torch.save(model.state_dict(), save_model_path)
            print(f"--> Best model saved to {save_model_path}")

print(f"Best Epoch: {best_epoch:03d}")
evaluate_utils.print_results(None, best_valid_res, best_test_res)