# generate_coldstart_dataset.py
import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm

def generate_coldstart_dataset(original_dir, coldstart_dir, cold_start_ratio=0.3):
    """
    生成物品冷啟動測試集
    
    Args:
        original_dir: 原始數據集目錄
        coldstart_dir: 冷啟動數據集目錄
        cold_start_ratio: 冷啟動物品比例（從測試集中選擇）
    """
    # 創建目錄
    os.makedirs(coldstart_dir, exist_ok=True)
    
    # ===== 1. 加載原始數據 =====
    print("Loading original data...")
    
    # 加載映射
    with open(os.path.join(original_dir, 'user_map.pkl'), 'rb') as f:
        user_map = pickle.load(f)
    
    with open(os.path.join(original_dir, 'item_map.pkl'), 'rb') as f:
        item_map = pickle.load(f)
    
    # 加載交互列表
    train_pairs = np.load(os.path.join(original_dir, 'train_list.npy'))
    valid_pairs = np.load(os.path.join(original_dir, 'valid_list.npy'))
    test_pairs = np.load(os.path.join(original_dir, 'test_list.npy'))
    
    # 加載物品文本
    with open(os.path.join(original_dir, 'item_text_list.txt'), 'r', encoding='utf-8') as f:
        item_texts = [line.strip() for line in f]
    
    # 加載嵌入（如果存在）
    embeddings_path = os.path.join(original_dir, 'embeddings', 'item_embeddings.npy')
    if os.path.exists(embeddings_path):
        item_embeddings = np.load(embeddings_path)
    else:
        item_embeddings = None
    
    print(f"Original dataset stats:")
    print(f"  Users: {len(user_map)}")
    print(f"  Items: {len(item_map)}")
    print(f"  Train interactions: {len(train_pairs)}")
    print(f"  Valid interactions: {len(valid_pairs)}")
    print(f"  Test interactions: {len(test_pairs)}")
    
    # ===== 2. 識別訓練集中的物品 =====
    print("\nIdentifying cold-start items...")
    
    # 訓練集中出現的物品
    train_items = set(train_pairs[:, 1])
    print(f"Items in training set: {len(train_items)}")
    
    # 測試集中出現的物品
    test_items = set(test_pairs[:, 1])
    print(f"Items in test set: {len(test_items)}")
    
    # 冷啟動物品：在測試集中但不在訓練集中
    cold_start_items = list(test_items - train_items)
    
    # 如果冷啟動物品太少，隨機選擇一些測試物品作為冷啟動
    if len(cold_start_items) < len(test_items) * cold_start_ratio:
        # 計算需要添加的冷啟動物品數量
        target_count = int(len(test_items) * cold_start_ratio)
        additional_needed = target_count - len(cold_start_items)
        
        if additional_needed > 0:
            # 從測試集的其他物品中隨機選擇
            other_test_items = list(test_items - set(cold_start_items))
            if additional_needed <= len(other_test_items):
                additional_cold = np.random.choice(other_test_items, additional_needed, replace=False)
                cold_start_items.extend(additional_cold)
                cold_start_items = list(set(cold_start_items))  # 去重
    
    print(f"Cold-start items identified: {len(cold_start_items)}")
    print(f"Cold-start ratio: {len(cold_start_items)/len(test_items):.2%}")
    
    # ===== 3. 篩選冷啟動測試集 =====
    print("\nFiltering cold-start test set...")
    
    # 測試集：只保留包含冷啟動物品的交互
    cold_start_test_mask = np.isin(test_pairs[:, 1], cold_start_items)
    cold_start_test_pairs = test_pairs[cold_start_test_mask]
    
    print(f"Original test interactions: {len(test_pairs)}")
    print(f"Cold-start test interactions: {len(cold_start_test_pairs)}")
    
    # ===== 4. 更新訓練集和驗證集 =====
    # 確保訓練集和驗證集不包含冷啟動物品
    print("\nUpdating training and validation sets...")
    
    # 訓練集：移除包含冷啟動物品的交互（雖然理論上應該沒有）
    train_mask = ~np.isin(train_pairs[:, 1], cold_start_items)
    new_train_pairs = train_pairs[train_mask]
    
    # 驗證集：移除包含冷啟動物品的交互
    valid_mask = ~np.isin(valid_pairs[:, 1], cold_start_items)
    new_valid_pairs = valid_pairs[valid_mask]
    
    print(f"Original train interactions: {len(train_pairs)}")
    print(f"New train interactions: {len(new_train_pairs)}")
    print(f"Removed from train: {len(train_pairs) - len(new_train_pairs)}")
    
    print(f"Original valid interactions: {len(valid_pairs)}")
    print(f"New valid interactions: {len(new_valid_pairs)}")
    print(f"Removed from valid: {len(valid_pairs) - len(new_valid_pairs)}")
    
    # ===== 5. 更新物品映射 =====
    print("\nUpdating item mapping...")
    
    # 找出所有出現的物品（訓練+驗證+冷啟動測試）
    all_items_in_use = set(
        list(new_train_pairs[:, 1]) + 
        list(new_valid_pairs[:, 1]) + 
        list(cold_start_test_pairs[:, 1])
    )
    
    # 創建新的物品映射
    new_item_map = {}
    reverse_original_item_map = {v: k for k, v in item_map.items()}
    
    for new_idx, old_item_id in enumerate(sorted(all_items_in_use)):
        original_asin = reverse_original_item_map[old_item_id]
        new_item_map[original_asin] = new_idx
    
    print(f"Original items: {len(item_map)}")
    print(f"New items after cold-start filtering: {len(new_item_map)}")
    print(f"Items removed: {len(item_map) - len(new_item_map)}")
    
    # ===== 6. 重新映射所有交互 =====
    print("\nRemapping interactions...")
    
    def remap_pairs(pairs, item_map_old_to_new):
        """將交互對中的物品ID重新映射"""
        remapped = []
        for uid, old_iid in pairs:
            original_asin = reverse_original_item_map[old_iid]
            if original_asin in new_item_map:  # 只保留存在的物品
                new_iid = new_item_map[original_asin]
                remapped.append([uid, new_iid])
        return np.array(remapped, dtype=np.int32)
    
    # 創建舊物品ID到新物品ID的映射
    old_to_new_item_map = {}
    for original_asin, new_iid in new_item_map.items():
        old_iid = item_map[original_asin]
        old_to_new_item_map[old_iid] = new_iid
    
    new_train_pairs_remapped = remap_pairs(new_train_pairs, old_to_new_item_map)
    new_valid_pairs_remapped = remap_pairs(new_valid_pairs, old_to_new_item_map)
    new_test_pairs_remapped = remap_pairs(cold_start_test_pairs, old_to_new_item_map)
    
    # ===== 7. 更新物品文本列表 =====
    print("\nUpdating item text list...")
    
    # 創建新物品文本列表
    new_item_texts = []
    for original_asin, new_iid in sorted(new_item_map.items(), key=lambda x: x[1]):
        old_iid = item_map[original_asin]
        new_item_texts.append(item_texts[old_iid])
    
    # ===== 8. 更新物品嵌入（如果存在） =====
    new_item_embeddings = None
    if item_embeddings is not None:
        print("Updating item embeddings...")
        new_item_embeddings = []
        for original_asin, new_iid in sorted(new_item_map.items(), key=lambda x: x[1]):
            old_iid = item_map[original_asin]
            new_item_embeddings.append(item_embeddings[old_iid])
        new_item_embeddings = np.array(new_item_embeddings)
    
    # ===== 9. 保存新數據集 =====
    print(f"\nSaving cold-start dataset to: {coldstart_dir}")
    
    # 保存交互列表
    np.save(os.path.join(coldstart_dir, 'train_list.npy'), new_train_pairs_remapped)
    np.save(os.path.join(coldstart_dir, 'valid_list.npy'), new_valid_pairs_remapped)
    np.save(os.path.join(coldstart_dir, 'test_list.npy'), new_test_pairs_remapped)
    
    # 保存映射
    with open(os.path.join(coldstart_dir, 'user_map.pkl'), 'wb') as f:
        pickle.dump(user_map, f)  # 用戶映射保持不變
    
    with open(os.path.join(coldstart_dir, 'item_map.pkl'), 'wb') as f:
        pickle.dump(new_item_map, f)
    
    # 保存物品文本
    with open(os.path.join(coldstart_dir, 'item_text_list.txt'), 'w', encoding='utf-8') as f:
        for text in new_item_texts:
            f.write(text + '\n')
    
    # 保存物品嵌入
    if new_item_embeddings is not None:
        embeddings_dir = os.path.join(coldstart_dir, 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        np.save(os.path.join(embeddings_dir, 'item_embeddings.npy'), new_item_embeddings)
    
    # ===== 10. 複製其他可能需要的文件 =====
    # 複製dataset_stats.json（如果存在）
    stats_path = os.path.join(original_dir, 'dataset_stats.json')
    if os.path.exists(stats_path):
        shutil.copy2(stats_path, os.path.join(coldstart_dir, 'dataset_stats.json'))
    
    # ===== 11. 生成冷啟動數據集統計信息 =====
    print("\n" + "="*50)
    print("COLD-START DATASET STATISTICS")
    print("="*50)
    
    # 計算冷啟動物品在測試集中的比例
    test_items_cold = set(new_test_pairs_remapped[:, 1])
    all_items_in_test = len(test_items_cold)
    
    # 找出在訓練集中沒出現的測試物品（真正的冷啟動）
    train_items_set = set(new_train_pairs_remapped[:, 1])
    true_cold_items = test_items_cold - train_items_set
    
    print(f"Users: {len(user_map)}")
    print(f"Items (total): {len(new_item_map)}")
    print(f"Items in training set: {len(train_items_set)}")
    print(f"Items in test set: {all_items_in_test}")
    print(f"True cold-start items in test set: {len(true_cold_items)}")
    print(f"Cold-start ratio (items): {len(true_cold_items)/all_items_in_test:.2%}")
    print(f"\nInteractions:")
    print(f"  Train: {len(new_train_pairs_remapped)}")
    print(f"  Valid: {len(new_valid_pairs_remapped)}")
    print(f"  Test (cold-start): {len(new_test_pairs_remapped)}")
    
    # 保存詳細統計
    cold_stats = {
        'n_users': len(user_map),
        'n_items': len(new_item_map),
        'n_items_train': len(train_items_set),
        'n_items_test': all_items_in_test,
        'n_cold_start_items': len(true_cold_items),
        'cold_start_ratio': len(true_cold_items) / all_items_in_test,
        'train_interactions': len(new_train_pairs_remapped),
        'valid_interactions': len(new_valid_pairs_remapped),
        'test_interactions': len(new_test_pairs_remapped),
        'cold_start_test_interactions': len(new_test_pairs_remapped)
    }
    
    with open(os.path.join(coldstart_dir, 'cold_start_stats.json'), 'w') as f:
        import json
        json.dump(cold_stats, f, indent=2)
    
    print(f"\n✅ Cold-start dataset generated successfully!")
    print(f"   Location: {coldstart_dir}")
    
    return cold_stats

if __name__ == "__main__":
    # 配置參數
    ORIGINAL_DIR = "../datasets/amazon-instruments"
    COLDSTART_DIR = "../datasets/amazon-instruments_coldstart"
    COLD_START_RATIO = 0.3  # 30%的測試物品是冷啟動物品
    
    # 生成冷啟動數據集
    stats = generate_coldstart_dataset(ORIGINAL_DIR, COLDSTART_DIR, COLD_START_RATIO)
    
    # 輸出用於驗證的示例
    print("\n" + "="*50)
    print("VERIFICATION SAMPLE")
    print("="*50)
    
    # 加載一些數據進行驗證
    test_pairs = np.load(os.path.join(COLDSTART_DIR, 'test_list.npy'))
    with open(os.path.join(COLDSTART_DIR, 'item_map.pkl'), 'rb') as f:
        item_map_cold = pickle.load(f)
    
    # 取前5個測試樣本
    print("\nFirst 5 test interactions (cold-start):")
    for i in range(min(5, len(test_pairs))):
        uid, iid = test_pairs[i]
        # 獲取原始ASIN
        reverse_map = {v: k for k, v in item_map_cold.items()}
        asin = reverse_map[iid]
        print(f"  User {uid} -> Item {iid} (ASIN: {asin})")