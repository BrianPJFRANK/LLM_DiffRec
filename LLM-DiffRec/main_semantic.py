# main_semantic.py
"""
Train a semantic-aware diffusion model for recommendation
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

# 添加路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import models.semantic_diffusion as gd
from models.SemanticDNN import SemanticDNN, DualStreamSemanticDNN
from utils import semantic_utils
from utils import evaluate_utils
from utils import data_utils

import random
random_seed = 1
torch.manual_seed(random_seed)  # cpu
torch.cuda.manual_seed(random_seed)  # gpu
np.random.seed(random_seed)  # numpy
random.seed(random_seed)  # random
torch.backends.cudnn.deterministic = True  # cudnn

def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# 參數解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='amazon-instruments', help='dataset name')
parser.add_argument('--data_path', type=str, default='../datasets/', help='data path')
parser.add_argument('--use_semantic', action='store_true', help='use semantic embeddings')
parser.add_argument('--model_type', type=str, default='semantic', choices=['semantic', 'dual', 'original'], 
                    help='model type: semantic, dual, or original')
parser.add_argument('--semantic_dim', type=int, default=768, help='semantic embedding dimension')
parser.add_argument('--semantic_proj_dim', type=int, default=128, help='semantic projection dimension')

# 原始參數
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA/NPU')
parser.add_argument('--gpu', type=str, default='0', help='gpu/npu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models_semantic/', help='save model path')
parser.add_argument('--log_name', type=str, default='log_semantic', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# 模型參數
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# 擴散參數
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep')

args = parser.parse_args()
print("Semantic Diffusion Args:", args)

# 設備設置
if args.cuda and torch.npu.is_available():
    device = torch.device(f"npu:{args.gpu}")
    print(f"Using NPU Device: npu:{args.gpu}")
elif args.cuda and torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using CUDA Device: cuda:{args.gpu}")
else:
    device = torch.device("cpu")
    print("Using CPU Device")

print("Starting time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# ===== 數據加載 =====
train_path = os.path.join(args.data_path, args.dataset, 'train_list.npy')
valid_path = os.path.join(args.data_path, args.dataset, 'valid_list.npy')
test_path = os.path.join(args.data_path, args.dataset, 'test_list.npy')

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(
    train_path, valid_path, test_path
)

train_dataset = data_utils.DataDiffusion(train_data)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                         pin_memory=True, shuffle=True, num_workers=0)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_data = train_data + valid_y_data
    tv_dataset = data_utils.DataDiffusion(tv_data)
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)

mask_tv = train_data + valid_y_data

print(f'Data loaded. Users: {n_user}, Items: {n_item}')

# ===== 語義處理器 =====
semantic_processor = None
if args.use_semantic:
    data_dir = os.path.join(args.data_path, args.dataset)
    semantic_processor = semantic_utils.SemanticProcessor(data_dir, device=device)
    
    if semantic_processor.item_embeddings is None:
        print("⚠️ Semantic embeddings not available, falling back to original model")
        args.use_semantic = False

# ===== 構建擴散模型 =====
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError(f"Unimplemented mean type {args.mean_type}")

diffusion = gd.SemanticGaussianDiffusion(
    mean_type, args.noise_schedule, args.noise_scale, 
    args.noise_min, args.noise_max, args.steps, device
).to(device)

# ===== 構建語義模型 =====
out_dims = eval(args.dims) + [n_item]
in_dims = out_dims[::-1]

if args.use_semantic:
    if args.model_type == 'semantic':
        model = SemanticDNN(
            in_dims, out_dims, args.emb_size, 
            semantic_dim=args.semantic_dim,
            semantic_proj_dim=args.semantic_proj_dim,
            time_type=args.time_type, 
            norm=args.norm,
            use_semantic=True
        ).to(device)
    elif args.model_type == 'dual':
        model = DualStreamSemanticDNN(
            in_dims, out_dims, args.emb_size,
            semantic_dim=args.semantic_dim,
            time_type=args.time_type,
            norm=args.norm
        ).to(device)
    else:
        # 後向兼容原始模型
        from models.DNN import DNN
        model = DNN(in_dims, out_dims, args.emb_size, 
                   time_type=args.time_type, norm=args.norm).to(device)
else:
    # 不使用語義，使用原始模型
    from models.DNN import DNN
    model = DNN(in_dims, out_dims, args.emb_size, 
               time_type=args.time_type, norm=args.norm).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 參數統計
param_num = sum([param.nelement() for param in model.parameters()])
print(f"Model parameters: {param_num:,}")
print(f"Using semantic: {args.use_semantic}")
print(f"Model type: {args.model_type}")

# ===== 評估函數（支持語義） =====
def evaluate_semantic(data_loader, data_te, mask_his, topN, semantic_processor=None):
    """
    評估函數，支持語義輸入
    """
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]
    
    predict_items = []
    target_items = []
    
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
            batch = batch.to(device)
            
            # 計算用戶語義向量
            user_semantic = None
            if semantic_processor is not None:
                user_semantic = semantic_processor.compute_user_semantic_simple(batch)
            
            # 預測
            if user_semantic is not None:
                prediction = diffusion.p_sample(model, batch, args.sampling_steps, 
                                               user_semantic, args.sampling_noise)
            else:
                prediction = diffusion.p_sample(model, batch, args.sampling_steps, 
                                               None, args.sampling_noise)
            
            # 屏蔽歷史交互
            prediction[his_data.nonzero()] = -np.inf
            
            # 獲取topK推薦
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)
    
    # 計算評估指標
    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
    
    return test_results

# ===== 訓練循環 =====
best_recall, best_epoch = -100, 0
best_results, best_test_results = None, None

print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 20:
        print('-' * 18)
        print('Exiting from training early')
        break
    
    model.train()
    start_time = time.time()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_count += 1
        
        # 計算用戶語義向量
        user_semantic = None
        if args.use_semantic and semantic_processor is not None:
            user_semantic = semantic_processor.compute_user_semantic_simple(batch)
        
        # 優化步驟
        optimizer.zero_grad()
        
        if user_semantic is not None:
            losses = diffusion.training_losses(model, batch, user_semantic, args.reweight)
        else:
            losses = diffusion.training_losses(model, batch, None, args.reweight)
        
        loss = losses["loss"].mean()
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
    
    # 定期評估
    if epoch % 5 == 0:
        print(f"\nEpoch {epoch:03d}, Avg Loss: {avg_loss:.4f}")
        
        # 驗證集評估
        valid_results = evaluate_semantic(
            test_loader, valid_y_data, train_data, eval(args.topN), semantic_processor
        )
        
        # 測試集評估
        if args.tst_w_val:
            test_results = evaluate_semantic(
                test_twv_loader, test_y_data, mask_tv, eval(args.topN), semantic_processor
            )
        else:
            test_results = evaluate_semantic(
                test_loader, test_y_data, mask_tv, eval(args.topN), semantic_processor
            )
        
        evaluate_utils.print_results(None, valid_results, test_results)
        
        # 保存最佳模型
        if valid_results[1][1] > best_recall:  # recall@20
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results
            
            # 保存模型
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            
            model_filename = (
                f"{args.dataset}_semantic_{args.model_type}_"
                f"lr{args.lr}_bs{args.batch_size}_steps{args.steps}_"
                f"epoch{epoch}_recall{best_recall:.4f}.pth"
            )
            
            torch.save(model, os.path.join(args.save_path, model_filename))
            print(f"✅ Model saved: {model_filename}")
    
    print(f"Epoch {epoch:03d} - Train Loss: {avg_loss:.4f} - "
          f"Time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")
    print('---' * 18)

# ===== 最終結果 =====
print('=' * 18)
print(f"Best Epoch: {best_epoch:03d}")
print("Best Validation Results:")
evaluate_utils.print_results(None, best_results, None)
print("Best Test Results:")
evaluate_utils.print_results(None, None, best_test_results)
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")