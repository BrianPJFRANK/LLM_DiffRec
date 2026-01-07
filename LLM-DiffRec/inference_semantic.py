# inference_semantic.py
"""
Inference script for semantic-aware diffusion recommender model
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import models.semantic_diffusion as gd
from utils import semantic_utils
import evaluate_utils
import data_utils

# 參數解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='amazon-instruments', help='dataset name')
parser.add_argument('--data_path', type=str, default='./datasets/', help='data path')
parser.add_argument('--model_path', type=str, required=True, help='path to trained model')
parser.add_argument('--use_semantic', action='store_true', help='use semantic embeddings')
parser.add_argument('--model_type', type=str, default='semantic', choices=['semantic', 'dual', 'original'], 
                    help='model type used in training')
parser.add_argument('--cold_start', action='store_true', help='evaluate on cold-start test set')

# 推理參數
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA/NPU')
parser.add_argument('--gpu', type=str, default='0', help='gpu/npu card ID')

# 擴散推理參數
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of forward process during inference')

# 語義參數
parser.add_argument('--semantic_dim', type=int, default=768, help='semantic embedding dimension')

args = parser.parse_args()
print("Semantic Inference Args:", args)

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

print("Starting inference at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# ===== 確定數據集路徑 =====
if args.cold_start:
    dataset_dir = os.path.join(args.data_path, f"{args.dataset}_coldstart")
    print(f"Using cold-start dataset: {dataset_dir}")
else:
    dataset_dir = os.path.join(args.data_path, args.dataset)

# ===== 數據加載 =====
train_path = os.path.join(dataset_dir, 'train_list.npy')
valid_path = os.path.join(dataset_dir, 'valid_list.npy')
test_path = os.path.join(dataset_dir, 'test_list.npy')

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(
    train_path, valid_path, test_path
)

train_dataset = data_utils.DataDiffusion(train_data)
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
    semantic_processor = semantic_utils.SemanticProcessor(dataset_dir, device=device)
    
    if semantic_processor.item_embeddings is None:
        print("⚠️ Semantic embeddings not available, disabling semantic mode")
        args.use_semantic = False

# ===== 加載模型 =====
print(f"Loading model from: {args.model_path}")
try:
    model = torch.load(args.model_path, map_location=device)
    model.eval()
    print("✅ Model loaded successfully")
    
    # 檢查模型類型
    model_class = model.__class__.__name__
    print(f"Model class: {model_class}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ===== 創建擴散實例 =====
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError(f"Unimplemented mean type {args.mean_type}")

# 擴散參數從模型文件名中解析或使用默認值
# 注意：這裡需要根據訓練時的參數設置，實際使用時可能需要傳遞更多參數
diffusion = gd.SemanticGaussianDiffusion(
    mean_type, 
    noise_schedule='linear-var',
    noise_scale=0.0001,
    noise_min=0.0005,
    noise_max=0.005,
    steps=5,
    device=device
).to(device)

print("Diffusion model ready")

# ===== 冷啟動物品識別 =====
cold_start_items = None
if args.cold_start and semantic_processor is not None:
    print("\nIdentifying cold-start items...")
    # 加載訓練交互
    train_pairs = np.load(train_path)
    cold_start_items = semantic_processor.get_cold_start_items(train_pairs)
    print(f"Cold-start items identified: {len(cold_start_items)}")

# ===== 評估函數 =====
def evaluate_with_semantic(data_loader, data_te, mask_his, topN, 
                          semantic_processor=None, cold_start_items=None):
    """
    評估函數，支持語義輸入和冷啟動分析
    """
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]
    
    predict_items = []
    target_items = []
    
    # 收集目標物品
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    # 冷啟動分析統計
    cold_start_stats = {
        'total_users': 0,
        'users_with_cold_items': 0,
        'total_recommendations': 0,
        'cold_item_recommendations': 0,
        'cold_hits': 0
    }
    
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
            
            # 冷啟動分析（如果啟用）
            if cold_start_items is not None:
                batch_start = batch_idx * args.batch_size
                batch_end = min(batch_start + len(batch), e_N)
                
                for i in range(len(indices)):
                    user_idx = batch_start + i
                    if user_idx < len(target_items):
                        # 檢查該用戶是否有冷啟動目標物品
                        user_targets = set(target_items[user_idx])
                        user_cold_targets = user_targets.intersection(cold_start_items)
                        
                        if user_cold_targets:
                            cold_start_stats['users_with_cold_items'] += 1
                            cold_start_stats['total_recommendations'] += topN[-1]
                            
                            # 檢查推薦中是否有冷啟動物品
                            user_recommendations = indices[i]
                            cold_recommended = set(user_recommendations).intersection(cold_start_items)
                            
                            if cold_recommended:
                                cold_start_stats['cold_item_recommendations'] += len(cold_recommended)
                                
                                # 檢查是否命中了冷啟動目標
                                cold_hits = cold_recommended.intersection(user_cold_targets)
                                if cold_hits:
                                    cold_start_stats['cold_hits'] += len(cold_hits)
                    
                    cold_start_stats['total_users'] += 1
    
    # 計算評估指標
    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
    
    # 冷啟動分析結果
    if cold_start_items is not None and cold_start_stats['total_users'] > 0:
        print("\n" + "="*50)
        print("COLD-START ANALYSIS")
        print("="*50)
        print(f"Total users: {cold_start_stats['total_users']}")
        print(f"Users with cold-start targets: {cold_start_stats['users_with_cold_items']}")
        print(f"Cold-start target ratio: {cold_start_stats['users_with_cold_items']/cold_start_stats['total_users']:.2%}")
        
        if cold_start_stats['total_recommendations'] > 0:
            print(f"Cold items in recommendations: {cold_start_stats['cold_item_recommendations']}")
            print(f"Cold recommendation ratio: {cold_start_stats['cold_item_recommendations']/cold_start_stats['total_recommendations']:.2%}")
            
            if cold_start_stats['users_with_cold_items'] > 0:
                print(f"Cold-start hits: {cold_start_stats['cold_hits']}")
                print(f"Users with cold-start hits: {cold_start_stats['cold_hits'] / cold_start_stats['users_with_cold_items']:.2%}")
    
    return test_results

# ===== 執行評估 =====
print("\n" + "="*50)
print("EVALUATION START")
print("="*50)

# 驗證集評估
print("\n[Validation Set Evaluation]")
valid_results = evaluate_with_semantic(
    test_loader, valid_y_data, train_data, eval(args.topN), 
    semantic_processor, cold_start_items if args.cold_start else None
)
evaluate_utils.print_results(None, valid_results, None)

# 測試集評估
print("\n[Test Set Evaluation]")
if args.tst_w_val:
    test_results = evaluate_with_semantic(
        test_twv_loader, test_y_data, mask_tv, eval(args.topN),
        semantic_processor, cold_start_items if args.cold_start else None
    )
else:
    test_results = evaluate_with_semantic(
        test_loader, test_y_data, mask_tv, eval(args.topN),
        semantic_processor, cold_start_items if args.cold_start else None
    )
evaluate_utils.print_results(None, None, test_results)

# ===== 可選：冷啟動物品相似度分析 =====
if args.cold_start and semantic_processor is not None and cold_start_items:
    print("\n" + "="*50)
    print("COLD-START ITEM SIMILARITY ANALYSIS")
    print("="*50)
    
    # 隨機選擇幾個冷啟動物品分析
    sample_cold_items = list(cold_start_items)[:5] if len(cold_start_items) > 5 else cold_start_items
    
    for item_id in sample_cold_items:
        similar_items, similarities = semantic_processor.get_similar_items(item_id, top_k=5)
        
        print(f"\nCold-start Item {item_id}:")
        print(f"  Top 5 similar items (based on semantic embeddings):")
        for sim_item, sim_score in zip(similar_items, similarities):
            print(f"    Item {sim_item}: similarity = {sim_score:.4f}")
        
        # 檢查這些相似物品是否在訓練集中
        train_items_set = set(np.load(train_path)[:, 1])
        in_train = [sim_item in train_items_set for sim_item in similar_items]
        print(f"  Similar items in training set: {sum(in_train)}/{len(in_train)}")

print("\n" + "="*50)
print("INFERENCE COMPLETED")
print("="*50)
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")