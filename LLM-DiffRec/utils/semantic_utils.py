# utils/semantic_utils.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm

class SemanticProcessor:
    """
    Semantic Processor: Responsible for loading item embeddings and calculating user semantic vectors
    """
    def __init__(self, data_path, device='cpu'):
        self.data_path = data_path
        self.device = device
        self.item_embeddings = None
        self.item_embedding_dim = None
        self.load_item_embeddings()
    
    def load_item_embeddings(self):
        """Load pre-trained item semantic embeddings"""
        emb_path = os.path.join(self.data_path, 'embeddings', 'item_embeddings.npy')
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
            self.item_embeddings = torch.FloatTensor(embeddings).to(self.device)
            self.item_embedding_dim = embeddings.shape[1]
            print(f"Loaded item embeddings: {embeddings.shape}")
            return True
        else:
            print("Warning: No item embeddings found at:", emb_path)
            self.item_embeddings = None
            self.item_embedding_dim = 0
            return False
    
    def compute_user_semantic_simple(self, user_interactions):
        """
        Simple average: User semantic vector = average of interacted item embeddings
        
        Args:
            user_interactions: [batch_size, n_items] User interaction matrix (0/1)
        
        Returns:
            user_semantic: [batch_size, embedding_dim] User semantic vector
        """
        if self.item_embeddings is None:
            return None
        
        # Convert to float
        user_interactions = user_interactions.float().to(self.device)
        
        # Calculate weighted sum: sum of interacted item embeddings for each user
        weighted_sum = torch.matmul(user_interactions, self.item_embeddings)
        
        # Calculate number of interactions (avoid division by zero)
        interaction_counts = user_interactions.sum(dim=1, keepdim=True)
        interaction_counts = torch.clamp(interaction_counts, min=1.0)
        
        # Average
        user_semantic = weighted_sum / interaction_counts
        
        return user_semantic
    
    def compute_user_semantic_attention(self, user_interactions, attention_weights=None):
        """
        Attention-weighted average: use attention mechanism to weight the importance of different items
        
        Args:
            user_interactions: [batch_size, n_items]
            attention_weights: [batch_size, n_items] Attention weights (optional)
        
        Returns:
            user_semantic: [batch_size, embedding_dim]
        """
        if self.item_embeddings is None:
            return None
        
        user_interactions = user_interactions.float().to(self.device)
        
        if attention_weights is None:
            # Use simple average
            return self.compute_user_semantic_simple(user_interactions)
        
        # Use attention weights
        attention_weights = attention_weights.to(self.device)
        
        # Only consider interacted items
        masked_weights = attention_weights * user_interactions
        
        # Normalize weights
        weight_sums = masked_weights.sum(dim=1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        normalized_weights = masked_weights / weight_sums
        
        # Weighted average
        user_semantic = torch.matmul(normalized_weights, self.item_embeddings)
        
        return user_semantic
    
    def compute_user_semantic_transformer(self, user_interactions, item_ids=None):
        """
        Use Transformer encoder to calculate user semantics (more complex but more powerful)
        
        Args:
            user_interactions: [batch_size, n_items]
            item_ids: Optional specific item ID list
        
        Returns:
            user_semantic: [batch_size, embedding_dim]
        """
        # This is a more advanced implementation that can be extended later
        # Currently use simple average
        return self.compute_user_semantic_simple(user_interactions)
    
    def get_cold_start_items(self, train_interactions, all_items=None):
        """
        Identify cold start items (items not present in the training set)
        
        Args:
            train_interactions: Training set interaction data
            all_items: All items list
        
        Returns:
            cold_start_items: Cold start item index list
        """
        if isinstance(train_interactions, np.ndarray):
            train_items = set(train_interactions[:, 1]) if train_interactions.shape[1] > 1 else set()
        elif torch.is_tensor(train_interactions):
            train_items = set(train_interactions[:, 1].cpu().numpy())
        else:
            # Assume it's a sparse matrix or dataset
            # Here we need to adjust based on the actual data type
            train_items = set()
            # Simple implementation: assume train_interactions is a list of items
            train_items = set(train_interactions)
        
        if all_items is None:
            # Get all items
            all_items = set(range(len(self.item_embeddings))) if self.item_embeddings is not None else set()
        
        cold_start_items = list(all_items - train_items)
        return cold_start_items
    
    def compute_item_similarity_matrix(self):
        """
        Calculate item similarity matrix (used for cold start recommendation)
        
        Returns:
            similarity_matrix: [n_items, n_items] Cosine similarity matrix
        """
        if self.item_embeddings is None:
            return None
        
        # Normalize embeddings
        embeddings_norm = F.normalize(self.item_embeddings, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        
        return similarity_matrix
    
    def get_similar_items(self, item_id, top_k=10):
        """
        Get the most similar items to a specified item
        
        Args:
            item_id: Item ID
            top_k: Number of similar items to return
        
        Returns:
            similar_items: List of similar item IDs
            similarities: List of similarity scores
        """
        if self.item_embeddings is None:
            return [], []
        
        # Calculate similarity
        item_embedding = self.item_embeddings[item_id:item_id+1]
        similarities = F.cosine_similarity(item_embedding, self.item_embeddings, dim=1)
        
        # Exclude self
        similarities[item_id] = -1
        
        # Get top_k
        top_similarities, top_indices = torch.topk(similarities, top_k)
        
        return top_indices.cpu().numpy(), top_similarities.cpu().numpy()