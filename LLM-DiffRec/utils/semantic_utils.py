# utils/semantic_utils.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm

class SemanticProcessor:
    """
    語義處理器：負責加載物品嵌入和計算用戶語義向量
    """
    def __init__(self, data_path, device='cpu'):
        self.data_path = data_path
        self.device = device
        self.item_embeddings = None
        self.item_embedding_dim = None
        self.load_item_embeddings()
    
    def load_item_embeddings(self):
        """加載預訓練的物品語義嵌入"""
        emb_path = os.path.join(self.data_path, 'embeddings', 'item_embeddings.npy')
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
            self.item_embeddings = torch.FloatTensor(embeddings).to(self.device)
            self.item_embedding_dim = embeddings.shape[1]
            print(f"✅ Loaded item embeddings: {embeddings.shape}")
            return True
        else:
            print("⚠️ Warning: No item embeddings found at:", emb_path)
            self.item_embeddings = None
            self.item_embedding_dim = 0
            return False
    
    def compute_user_semantic_simple(self, user_interactions):
        """
        簡單平均：用戶語義向量 = 交互物品嵌入的平均
        
        Args:
            user_interactions: [batch_size, n_items] 用戶交互矩陣（0/1）
        
        Returns:
            user_semantic: [batch_size, embedding_dim] 用戶語義向量
        """
        if self.item_embeddings is None:
            return None
        
        # 轉換為浮點數
        user_interactions = user_interactions.float().to(self.device)
        
        # 計算加權和：每個用戶的交互物品嵌入之和
        weighted_sum = torch.matmul(user_interactions, self.item_embeddings)
        
        # 計算交互數量（避免除以零）
        interaction_counts = user_interactions.sum(dim=1, keepdim=True)
        interaction_counts = torch.clamp(interaction_counts, min=1.0)
        
        # 平均
        user_semantic = weighted_sum / interaction_counts
        
        return user_semantic
    
    def compute_user_semantic_attention(self, user_interactions, attention_weights=None):
        """
        注意力加權平均：使用注意力機制加權不同物品的重要性
        
        Args:
            user_interactions: [batch_size, n_items]
            attention_weights: [batch_size, n_items] 注意力權重（可選）
        
        Returns:
            user_semantic: [batch_size, embedding_dim]
        """
        if self.item_embeddings is None:
            return None
        
        user_interactions = user_interactions.float().to(self.device)
        
        if attention_weights is None:
            # 使用簡單平均
            return self.compute_user_semantic_simple(user_interactions)
        
        # 使用注意力權重
        attention_weights = attention_weights.to(self.device)
        
        # 只考慮有交互的物品
        masked_weights = attention_weights * user_interactions
        
        # 歸一化權重
        weight_sums = masked_weights.sum(dim=1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        normalized_weights = masked_weights / weight_sums
        
        # 加權平均
        user_semantic = torch.matmul(normalized_weights, self.item_embeddings)
        
        return user_semantic
    
    def compute_user_semantic_transformer(self, user_interactions, item_ids=None):
        """
        使用Transformer編碼器計算用戶語義（更複雜但更強大）
        
        Args:
            user_interactions: [batch_size, n_items]
            item_ids: 可選的具體物品ID列表
        
        Returns:
            user_semantic: [batch_size, embedding_dim]
        """
        # 這是一個更高級的實現，可以後續擴展
        # 目前先使用簡單平均
        return self.compute_user_semantic_simple(user_interactions)
    
    def get_cold_start_items(self, train_interactions, all_items=None):
        """
        識別冷啟動物品（在訓練集中未出現的物品）
        
        Args:
            train_interactions: 訓練集交互數據
            all_items: 所有物品列表
        
        Returns:
            cold_start_items: 冷啟動物品索引列表
        """
        if isinstance(train_interactions, np.ndarray):
            train_items = set(train_interactions[:, 1]) if train_interactions.shape[1] > 1 else set()
        elif torch.is_tensor(train_interactions):
            train_items = set(train_interactions[:, 1].cpu().numpy())
        else:
            # 假設是稀疏矩陣或數據集
            # 這裡需要根據實際數據類型調整
            train_items = set()
            # 簡單實現：假設train_interactions是物品列表
            train_items = set(train_interactions)
        
        if all_items is None:
            # 獲取所有物品
            all_items = set(range(len(self.item_embeddings))) if self.item_embeddings is not None else set()
        
        cold_start_items = list(all_items - train_items)
        return cold_start_items
    
    def compute_item_similarity_matrix(self):
        """
        計算物品相似度矩陣（用於冷啟動推薦）
        
        Returns:
            similarity_matrix: [n_items, n_items] 餘弦相似度矩陣
        """
        if self.item_embeddings is None:
            return None
        
        # 歸一化嵌入
        embeddings_norm = F.normalize(self.item_embeddings, p=2, dim=1)
        
        # 計算相似度矩陣
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        
        return similarity_matrix
    
    def get_similar_items(self, item_id, top_k=10):
        """
        獲取與指定物品最相似的物品
        
        Args:
            item_id: 物品ID
            top_k: 返回的相似物品數量
        
        Returns:
            similar_items: 相似物品ID列表
            similarities: 相似度分數列表
        """
        if self.item_embeddings is None:
            return [], []
        
        # 計算相似度
        item_embedding = self.item_embeddings[item_id:item_id+1]
        similarities = F.cosine_similarity(item_embedding, self.item_embeddings, dim=1)
        
        # 排除自身
        similarities[item_id] = -1
        
        # 獲取top_k
        top_similarities, top_indices = torch.topk(similarities, top_k)
        
        return top_indices.cpu().numpy(), top_similarities.cpu().numpy()


def create_semantic_aware_dataset(original_dataset, cold_start_ratio=0.3):
    """
    創建語義感知的數據集（包含冷啟動物品）
    """
    # 這是一個高級功能，可以後續實現
    pass