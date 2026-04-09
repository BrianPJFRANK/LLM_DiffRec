import os
import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# 1. Dynamically add the root directory (LLM-DiffRec) to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

# Import necessary models and utils from root
from models.LightGCN import LightGCN
from models.SemanticDNN import FiLMSemanticDNN
import models.semantic_diffusion as gd
from utils import evaluate_utils, data_utils, semantic_utils

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
DATA_DIR = os.path.join(ROOT_DIR, '../datasets/amazon-instruments')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'paper_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


LGCN_MODEL_PATH = os.path.join(ROOT_DIR, 'saved_models_baselines/lightgcn_L3_best.pth')
FILM_MODEL_PATH = os.path.join(ROOT_DIR, 'saved_models_semantic/amazon-instruments_semantic_film_lr0.00025_bs128_steps10_epoch10_recall0.1369.pth')

# Academic styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'pdf.fonttype': 42})

def build_adj_mat(R, n_users, n_items, device):
    """Build normalized adjacency matrix for LightGCN"""
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()

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

def load_and_split_users():
    print("Loading data for sparsity analysis...")
    train_path = os.path.join(DATA_DIR, 'train_list.npy')
    valid_path = os.path.join(DATA_DIR, 'valid_list.npy')
    test_path  = os.path.join(DATA_DIR, 'test_list.npy')

    train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)

    user_degrees = np.array(train_data.sum(axis=1)).flatten()

    cold_users   = np.where((user_degrees >= 1) & (user_degrees <= 5))[0]
    normal_users = np.where((user_degrees > 5)  & (user_degrees <= 15))[0]
    active_users = np.where(user_degrees > 15)[0]

    print(f"Total Users: {n_user}")
    print(f"  Cold Bucket (1-5):    {len(cold_users)} users")
    print(f"  Normal Bucket (6-15): {len(normal_users)} users")
    print(f"  Active Bucket (>15):  {len(active_users)} users")

    buckets = {
        'Cold (1-5)': cold_users,
        'Normal (6-15)': normal_users,
        'Active (>15)': active_users
    }

    mask_tv = train_data + valid_y_data
    return train_data, buckets, test_y_data, mask_tv, n_user, n_item

def evaluate_bucket_lightgcn(model_path, train_data, buckets, test_y_data, mask_tv, n_user, n_item, device):
    print(f"Evaluating LightGCN on buckets...")

    model = LightGCN(n_user, n_item, emb_dim=64, n_layers=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    norm_adj_tensor = build_adj_mat(train_data, n_user, n_item, device=torch.device('cpu'))

    with torch.no_grad():
        all_users, all_items = model(norm_adj_tensor)
        logits = torch.matmul(all_users, all_items.T)
        logits[mask_tv.toarray() > 0] = -np.inf
        _, all_indices = torch.topk(logits, 20)
        all_indices = all_indices.cpu().numpy()

    results = {}
    for bucket_name, user_indices in buckets.items():
        target_items = []
        predict_items = []
        for u in user_indices:
            ground_truth = test_y_data[u, :].nonzero()[1].tolist()
            if len(ground_truth) > 0:
                target_items.append(ground_truth)
                predict_items.append(all_indices[u].tolist())

        if len(target_items) > 0:
            res = evaluate_utils.computeTopNAccuracy(target_items, predict_items, [20])
            results[bucket_name] = res[1][0] # Recall@20
        else:
            results[bucket_name] = 0.0

    return results

def evaluate_bucket_film(model_path, train_data, buckets, test_y_data, mask_tv, n_user, n_item, device):
    print(f"Evaluating FiLM on buckets...")

    # 1. Init Semantic Processor
    semantic_processor = semantic_utils.SemanticProcessor(DATA_DIR, device=device)
    item_embs = semantic_processor.item_embeddings

    # 2. Init Model Architectures (Hardcoded from best args)
    # dims = [200, 600]
    out_dims = [200, 600, n_item]
    in_dims = out_dims[::-1]

    print("Loading FiLM model weights...")
    try:
        # main_semantic.py 第 282 行使用的是 torch.save(model, ...) 完整保存
        model = torch.load(model_path, map_location=device)
    except:
        # 兼容模式：如果使用的是 load_state_dict
        model = FiLMSemanticDNN(
            in_dims, out_dims, emb_size=10,
            semantic_dim=1024, semantic_hidden_dim=256,
            time_type='cat', norm=False, use_semantic=True
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # 3. Init Diffusion Process (Hardcoded from best args)
    diffusion = gd.SemanticGaussianDiffusion(
        mean_type=gd.ModelMeanType.START_X, # 'x0'
        noise_schedule='linear-var',
        noise_scale=0.0012,
        noise_min=0.006,
        noise_max=0.025,
        steps=10,                           # 'steps=10'
        device=device
    ).to(device)

    # 4. Batch Inference (To avoid OOM)
    train_dataset = data_utils.DataDiffusion(train_data)
    test_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    all_indices_list = []

    print("Generating predictions via Diffusion p_sample...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            start = batch_idx * 128
            end = min((batch_idx + 1) * 128, n_user)

            # Compute semantic embeddings for the batch
            user_semantic = semantic_processor.compute_user_semantic_simple(batch)

            # Diffusion Sampling
            prediction = diffusion.p_sample(
                model, batch, 0,
                user_semantic=user_semantic,
                item_embeddings=item_embs,
                sampling_noise=False
            )

            # Mask history and valid
            prediction[mask_tv[start:end].toarray() > 0] = -np.inf

            _, indices = torch.topk(prediction, 20)
            all_indices_list.append(indices.cpu().numpy())

    all_indices = np.concatenate(all_indices_list, axis=0)

    # 5. Bucket Evaluation
    results = {}
    for bucket_name, user_indices in buckets.items():
        target_items = []
        predict_items = []
        for u in user_indices:
            ground_truth = test_y_data[u, :].nonzero()[1].tolist()
            if len(ground_truth) > 0:
                target_items.append(ground_truth)
                predict_items.append(all_indices[u].tolist())

        if len(target_items) > 0:
            res = evaluate_utils.computeTopNAccuracy(target_items, predict_items, [20])
            results[bucket_name] = res[1][0] # Recall@20
        else:
            results[bucket_name] = 0.0

    return results

def plot_sparsity_results(results_lgcn, results_film):
    print("Generating Sparsity Analysis Chart...")
    groups = list(results_lgcn.keys())

    scores_lgcn = [results_lgcn[g] for g in groups]
    scores_film = [results_film[g] for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    rects1 = ax.bar(x - width/2, scores_lgcn, width, label='LightGCN', color='#8C8C8C', edgecolor='black')
    rects2 = ax.bar(x + width/2, scores_film, width, label='FiLM (Ours)', color='#C44E52', edgecolor='black')

    ax.set_ylabel('Recall@20', fontweight='bold')
    ax.set_title('Performance across User Sparsity Levels (amazon-instruments)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'sparsity_analysis.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"--> Saved {save_path}")

if __name__ == "__main__":
    device = torch.device("npu:0" if hasattr(torch, 'npu') and torch.npu.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, buckets, test_y, mask, n_user, n_item = load_and_split_users()

    res_lgcn = evaluate_bucket_lightgcn(LGCN_MODEL_PATH, train_data, buckets, test_y, mask, n_user, n_item, device)
    res_film = evaluate_bucket_film(FILM_MODEL_PATH, train_data, buckets, test_y, mask, n_user, n_item, device)

    plot_sparsity_results(res_lgcn, res_film)
    print("========================================")
    print("Sparsity analysis complete.")