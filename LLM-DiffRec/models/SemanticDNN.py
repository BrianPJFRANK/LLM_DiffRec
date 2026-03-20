# models/SemanticDNN.py
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SemanticDNN(nn.Module):
    """
    支持語義輸入的深度神經網路，用於反向擴散過程。
    融合三種信息：
    1. 用戶交互向量 x_t
    2. 時間嵌入
    3. 用戶語義向量（從物品語義嵌入聚合得到）
    """
    def __init__(self, in_dims, out_dims, emb_size, semantic_dim=768, 
                 time_type="cat", norm=False, dropout=0.5, 
                 semantic_proj_dim=128, use_semantic=True):
        super(SemanticDNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.semantic_dim = semantic_dim
        self.semantic_proj_dim = semantic_proj_dim
        self.use_semantic = use_semantic
        self.norm = norm

        # 時間嵌入層
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 語義投影層：將原始語義向量降維
        if self.use_semantic:
            self.semantic_projection = nn.Sequential(
                nn.Linear(semantic_dim, semantic_proj_dim),
                nn.ReLU(),
                nn.Linear(semantic_proj_dim, semantic_proj_dim),
                nn.Tanh()
            )
        else:
            self.semantic_projection = None
        
        # 根據是否使用語義調整輸入維度
        if self.time_type == "cat":
            if self.use_semantic:
                # 輸入維度：交互向量 + 時間嵌入 + 語義投影
                additional_dims = self.time_emb_dim + semantic_proj_dim
            else:
                # 後向兼容：交互向量 + 時間嵌入
                additional_dims = self.time_emb_dim
            
            in_dims_temp = [self.in_dims[0] + additional_dims] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        
        out_dims_temp = self.out_dims
        
        # 輸入層
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        # 輸出層
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """Xavier初始化權重"""
        layers = []
        if hasattr(self, 'in_layers'):
            layers.extend(self.in_layers)
        if hasattr(self, 'out_layers'):
            layers.extend(self.out_layers)
        if hasattr(self, 'emb_layer'):
            layers.append(self.emb_layer)
        if hasattr(self, 'semantic_projection') and self.semantic_projection is not None:
            layers.extend(list(self.semantic_projection))
        
        for layer in layers:
            if isinstance(layer, nn.Linear):
                # Xavier Initialization
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, user_semantic=None):
        """
        Args:
            x: 用戶交互向量 [batch_size, n_items]
            timesteps: 時間步 [batch_size]
            user_semantic: 用戶語義向量 [batch_size, semantic_dim] 或 None
        
        Returns:
            h: 預測的交互概率 [batch_size, n_items]
        """
        # 時間嵌入
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        
        # 可選的輸入歸一化
        if self.norm:
            x = F.normalize(x)
        
        x = self.drop(x)
        
        # 處理語義輸入
        if self.use_semantic and user_semantic is not None:
            # 投影語義向量
            semantic_proj = self.semantic_projection(user_semantic)
            # 拼接所有輸入
            h = torch.cat([x, time_emb, semantic_proj], dim=-1)
        else:
            # 後向兼容：如果不使用語義或沒有語義輸入
            if self.use_semantic:
                # 創建零向量作為語義輸入
                batch_size = x.shape[0]
                semantic_proj = torch.zeros(batch_size, self.semantic_proj_dim).to(x.device)
                h = torch.cat([x, time_emb, semantic_proj], dim=-1)
            else:
                # 原始模式：只拼接交互向量和時間嵌入
                h = torch.cat([x, time_emb], dim=-1)
        
        # 前向傳播：輸入層
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        # 前向傳播：輸出層
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h


class DualStreamSemanticDNN(nn.Module):
    """
    雙流架構的語義DNN（策略C的實現）
    兩個流：
    1. 交互流：處理用戶交互向量
    2. 語義流：處理用戶語義向量
    最後融合兩個流的輸出
    """
    def __init__(self, in_dims, out_dims, emb_size, semantic_dim=768,
                 time_type="cat", norm=False, dropout=0.5,
                 interaction_hidden=512, semantic_hidden=256,
                 fusion_hidden=512):
        super(DualStreamSemanticDNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.semantic_dim = semantic_dim
        self.norm = norm
        
        # 時間嵌入層
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 交互流：處理用戶交互向量 + 時間嵌入
        if self.time_type == "cat":
            interaction_input_dim = in_dims[0] + self.time_emb_dim
        else:
            interaction_input_dim = in_dims[0]
        
        self.interaction_stream = nn.Sequential(
            nn.Linear(interaction_input_dim, interaction_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_hidden, interaction_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_hidden // 2, fusion_hidden // 2)
        )
        
        # 語義流：處理用戶語義向量
        self.semantic_stream = nn.Sequential(
            nn.Linear(semantic_dim, semantic_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden, semantic_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden // 2, fusion_hidden // 2)
        )
        
        # 融合層：合併兩個流的輸出
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, out_dims[-1])
        )
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """初始化權重"""
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, 0, 0.001)
        
        # 初始化交互流
        for layer in self.interaction_stream:
            init_layer(layer)
        
        # 初始化語義流
        for layer in self.semantic_stream:
            init_layer(layer)
        
        # 初始化融合層
        for layer in self.fusion_layers:
            init_layer(layer)
        
        # 初始化時間嵌入層
        init_layer(self.emb_layer)
    
    def forward(self, x, timesteps, user_semantic=None):
        # 時間嵌入
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        
        if self.norm:
            x = F.normalize(x)
        
        x = self.drop(x)
        
        # 交互流輸入
        if self.time_type == "cat":
            interaction_input = torch.cat([x, time_emb], dim=-1)
        else:
            interaction_input = x
        
        # 交互流
        interaction_output = self.interaction_stream(interaction_input)
        
        # 語義流
        if user_semantic is not None:
            semantic_output = self.semantic_stream(user_semantic)
        else:
            # 如果沒有語義輸入，使用零向量
            batch_size = x.shape[0]
            semantic_output = torch.zeros(batch_size, 
                                         self.interaction_stream[-1].out_features).to(x.device)
        
        # 融合兩個流的輸出
        fused = torch.cat([interaction_output, semantic_output], dim=-1)
        
        # 融合層
        output = self.fusion_layers(fused)
        
        return output

class FiLMSemanticDNN(nn.Module):
    """
    基於 FiLM (Feature-wise Linear Modulation) 的語義擴散神經網路。
    將使用者語義 $s_u$ 作為條件控制信號，對去噪主幹網路的隱藏特徵進行仿射變換。
    """
    def __init__(self, in_dims, out_dims, emb_size, semantic_dim=768, 
                 time_type="cat", norm=False, dropout=0.5, 
                 semantic_hidden_dim=256, use_semantic=True):
        super(FiLMSemanticDNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.semantic_dim = semantic_dim
        self.use_semantic = use_semantic
        self.norm = norm

        # 時間嵌入層
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 1. 定義去噪主幹網路 (Denoising Backbone) 的維度
        # 輸入不再包含語義向量，僅拼接交互向量 x_t 和時間嵌入 tau(t)
        if self.time_type == "cat":
            backbone_input_dim = self.in_dims[0] + self.time_emb_dim
            in_dims_temp = [backbone_input_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        
        out_dims_temp = self.out_dims
        
        # 主幹網路 - 輸入層
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        # 主幹網路 - 輸出層
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        
        # 2. 定義語義控制器 (Semantic Controller)
        # 目標是主幹網路的第一個隱藏層 (即 in_dims_temp[1])
        self.film_target_dim = in_dims_temp[1] 
        
        if self.use_semantic:
            # 輸出維度為 2 * target_dim，用來切分為 gamma (縮放) 和 beta (平移)
            self.semantic_controller = nn.Sequential(
                nn.Linear(self.semantic_dim, semantic_hidden_dim),
                nn.ReLU(),
                nn.Linear(semantic_hidden_dim, 2 * self.film_target_dim)
            )
        else:
            self.semantic_controller = None
            
        # 執行權重初始化 (包含核心的 Zero-Initialization)
        self.init_weights()
    
    def init_weights(self):
        """初始化權重，嚴格遵循 SemanticDNN 的風格與 FiLM 的防崩潰約束"""
        # --- 主幹網路與時間嵌入的 Xavier 初始化 ---
        layers_to_init = []
        if hasattr(self, 'in_layers'):
            layers_to_init.extend(self.in_layers)
        if hasattr(self, 'out_layers'):
            layers_to_init.extend(self.out_layers)
        if hasattr(self, 'emb_layer'):
            layers_to_init.append(self.emb_layer)
            
        for layer in layers_to_init:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                if layer.bias is not None:
                    layer.bias.data.normal_(0.0, 0.001)

        # --- 語義控制器 (Semantic Controller) 的特殊初始化 ---
        if self.use_semantic and self.semantic_controller is not None:
            # 前面的特徵提取層使用常規 Xavier 初始化
            for layer in self.semantic_controller[:-1]:
                if isinstance(layer, nn.Linear):
                    size = layer.weight.size()
                    fan_out = size[0]
                    fan_in = size[1]
                    std = np.sqrt(2.0 / (fan_in + fan_out))
                    layer.weight.data.normal_(0.0, std)
                    if layer.bias is not None:
                        layer.bias.data.normal_(0.0, 0.001)
            
            # 極度重要：最後一層的權重與偏差必須初始化為 0
            # 確保初始輸出 gamma=0, beta=0，使 FiLM 層等同於 Identity Mapping
            final_layer = self.semantic_controller[-1]
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)
    
    def forward(self, x, timesteps, user_semantic=None):
        # 1. 時間嵌入
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        
        # 2. 構建主幹網路的輸入 (僅拼接 x_t 和時間嵌入)
        if self.time_type == "cat":
            h = torch.cat([x, time_emb], dim=-1)
        else:
            h = x
            
        # 3. 前向傳播：通過第一層隱藏層並激活
        h = self.in_layers[0](h)
        h = torch.tanh(h)
        
        # 4. ======= FiLM 語義調變 (Semantic Modulation) =======
        if self.use_semantic and user_semantic is not None:
            # 計算調變參數
            film_params = self.semantic_controller(user_semantic) # [batch_size, 2 * film_target_dim]
            
            # 切分為縮放因子 gamma 和平移因子 beta
            gamma, beta = torch.chunk(film_params, 2, dim=-1) # 各自 [batch_size, film_target_dim]
            
            # 執行調變: h_new = (1 + gamma) * h + beta
            h = (1.0 + gamma) * h + beta

        # 5. 繼續主幹網路的剩餘前向傳播
        for i in range(1, len(self.in_layers)):
            h = self.in_layers[i](h)
            h = torch.tanh(h)
            
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
                
        return h

class FiLMDotProductDNN(nn.Module):
    """
    基於 FiLM 且輸出層為「語義內積」的擴散神經網路 (專為冷啟動設計)。
    不再依賴 ID 映射，而是預測用戶的「理想物品語義」，並與全局物品語義庫進行內積打分。
    """
    def __init__(self, in_dims, out_dims, emb_size, semantic_dim=1024,
                 time_type="cat", norm=False, dropout=0.5,
                 semantic_hidden_dim=256, use_semantic=True):
        super(FiLMDotProductDNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.n_items = out_dims[-1]  # 最後一個維度即為物品總數
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.semantic_dim = semantic_dim
        self.use_semantic = use_semantic
        self.norm = norm

        # 時間嵌入層
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # 1. 定義去噪主幹網路的維度 (剔除最後的 n_items)
        if self.time_type == "cat":
            backbone_input_dim = self.in_dims[0] + self.time_emb_dim
            in_dims_temp = [backbone_input_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)

        # 這裡 out_dims_temp 捨棄最後的 n_items，僅保留隱藏層維度
        out_dims_temp = self.out_dims[:-1]

        # 主幹網路 - 輸入層
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])

        # 主幹網路 - 輸出層 (僅到最後一個隱藏層，例如 400)
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)

        # 2. 定義 FiLM 語義控制器
        self.film_target_dim = in_dims_temp[1]
        if self.use_semantic:
            self.semantic_controller = nn.Sequential(
                nn.Linear(self.semantic_dim, semantic_hidden_dim),
                nn.ReLU(),
                nn.Linear(semantic_hidden_dim, 2 * self.film_target_dim)
            )
        else:
            self.semantic_controller = None

        # 3. 💥 全新：語義內積輸出層 💥
        # 將最後的隱藏層映射回 1024 維的語義空間
        last_hidden_dim = out_dims_temp[-1] if len(out_dims_temp) > 0 else in_dims_temp[-1]
        self.semantic_output_layer = nn.Linear(last_hidden_dim, self.semantic_dim)

        # 為了穩定訓練，加入可學習的溫度縮放 (Temperature) 與 物品偏差 (Item Bias)
        self.tau = nn.Parameter(torch.ones(1) * 10.0) # 初始放大 10 倍，避免 L2 內積數值過小
        #self.item_bias = nn.Parameter(torch.zeros(self.n_items))

        self.init_weights()

    def init_weights(self):
        """初始化權重，包含 FiLM 的零初始化與常規 Xavier"""
        layers_to_init = []
        if hasattr(self, 'in_layers'): layers_to_init.extend(self.in_layers)
        if hasattr(self, 'out_layers'): layers_to_init.extend(self.out_layers)
        layers_to_init.extend([self.emb_layer, self.semantic_output_layer])

        for layer in layers_to_init:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, 0, 0.001)

        # FiLM 控制器的零初始化防崩潰
        if self.use_semantic and self.semantic_controller is not None:
            for layer in self.semantic_controller[:-1]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.normal_(layer.bias, 0, 0.001)
            nn.init.zeros_(self.semantic_controller[-1].weight)
            if self.semantic_controller[-1].bias is not None:
                nn.init.zeros_(self.semantic_controller[-1].bias)

    def forward(self, x, timesteps, user_semantic=None, item_embeddings=None):
        """
        新增 item_embeddings 參數 [n_items, semantic_dim]
        """
        # 1. 時間嵌入與輸入處理
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        if self.norm: x = F.normalize(x)
        x = self.drop(x)

        h = torch.cat([x, time_emb], dim=-1) if self.time_type == "cat" else x

        # 2. 第一層與 FiLM 調變
        h = self.in_layers[0](h)
        h = torch.tanh(h)
        if self.use_semantic and user_semantic is not None:
            film_params = self.semantic_controller(user_semantic)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            h = (1.0 + gamma) * h + beta

        # 3. 主幹網路前向傳播
        for i in range(1, len(self.in_layers)):
            h = self.in_layers[i](h)
            h = torch.tanh(h)
        for layer in self.out_layers:
            h = layer(h)
            h = torch.tanh(h)

        # 4. 💥 語義預測與內積打分 💥
        # 推測用戶此刻理想的物品語義 [batch_size, 1024]
        pred_semantic = self.semantic_output_layer(h)

        if item_embeddings is None:
            raise ValueError("item_embeddings MUST be provided for FiLMDotProductDNN")

        # L2 正規化 (防止維度爆炸)
        pred_semantic_norm = F.normalize(pred_semantic, p=2, dim=-1)
        item_embeddings_norm = F.normalize(item_embeddings, p=2, dim=-1)

        # 矩陣內積: [batch_size, 1024] x [1024, n_items] -> [batch_size, n_items]
        logits = torch.matmul(pred_semantic_norm, item_embeddings_norm.transpose(0, 1))

        # 縮放與加上偏差，還原到 DiffRec 適應的 Logits 空間
        out = logits * self.tau
        #out = logits * self.tau + self.item_bias

        return out