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
    Deep Neural Network supporting semantic input for the reverse diffusion process.
    Fuses three types of information:
    1. User interaction vector x_t
    2. Timestep embedding
    3. User semantic vector (aggregated from item semantic embeddings)
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

        # Timestep embedding layer
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # Semantic projection layer: reduce dimensionality of original semantic vector
        if self.use_semantic:
            self.semantic_projection = nn.Sequential(
                nn.Linear(semantic_dim, semantic_proj_dim),
                nn.ReLU(),
                nn.Linear(semantic_proj_dim, semantic_proj_dim),
                nn.Tanh()
            )
        else:
            self.semantic_projection = None
        
        # Adjust input dimension based on whether semantics are used
        if self.time_type == "cat":
            if self.use_semantic:
                # Input dimensions: interaction vector + timestep embedding + semantic projection
                additional_dims = self.time_emb_dim + semantic_proj_dim
            else:
                # Backward compatibility: interaction vector + timestep embedding
                additional_dims = self.time_emb_dim
            
            in_dims_temp = [self.in_dims[0] + additional_dims] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        
        out_dims_temp = self.out_dims
        
        # Input layers
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        # Output layers
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """Xavier initialization for weights"""
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
    
    def forward(self, x, timesteps, user_semantic=None, item_embeddings=None):
        """
        Args:
            x: User interaction vector [batch_size, n_items]
            timesteps: Timesteps [batch_size]
            user_semantic: User semantic vector [batch_size, semantic_dim] or None
        
        Returns:
            h: Predicted interaction probabilities [batch_size, n_items]
        """
        # Timestep embedding
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        
        # Optional input normalization
        if self.norm:
            x = F.normalize(x)
        
        x = self.drop(x)
        
        # Process semantic input
        if self.use_semantic and user_semantic is not None:
            # Project semantic vector
            semantic_proj = self.semantic_projection(user_semantic)
            # Concatenate all inputs
            h = torch.cat([x, time_emb, semantic_proj], dim=-1)
        else:
            # Backward compatibility: if semantics are not used or no semantic input
            if self.use_semantic:
                # Create zero vector as semantic input
                batch_size = x.shape[0]
                semantic_proj = torch.zeros(batch_size, self.semantic_proj_dim).to(x.device)
                h = torch.cat([x, time_emb, semantic_proj], dim=-1)
            else:
                # Original mode: only concatenate interaction vector and timestep embedding
                h = torch.cat([x, time_emb], dim=-1)
        
        # Forward pass: input layers
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        # Forward pass: output layers
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h


class DualStreamSemanticDNN(nn.Module):
    """
    Dual-stream architecture semantic DNN (Implementation of Strategy C)
    Two streams:
    1. Interaction stream: process user interaction vector
    2. Semantic stream: process user semantic vector
    Finally, fuse the outputs of the two streams
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
        
        # Timestep embedding layer
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # Interaction stream: process user interaction vector + timestep embedding
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
        
        # Semantic stream: process user semantic vector
        self.semantic_stream = nn.Sequential(
            nn.Linear(semantic_dim, semantic_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden, semantic_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden // 2, fusion_hidden // 2)
        )
        
        # Fusion layer: merge the outputs of the two streams
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, out_dims[-1])
        )
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, 0, 0.001)
        
        # Initialize interaction stream
        for layer in self.interaction_stream:
            init_layer(layer)
        
        # Initialize semantic stream
        for layer in self.semantic_stream:
            init_layer(layer)
        
        # Initialize fusion layer
        for layer in self.fusion_layers:
            init_layer(layer)
        
        # Initialize timestep embedding layer
        init_layer(self.emb_layer)
    
    def forward(self, x, timesteps, user_semantic=None, item_embeddings=None):
        # Timestep embedding
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        
        if self.norm:
            x = F.normalize(x)
        
        x = self.drop(x)
        
        # Interaction stream input
        if self.time_type == "cat":
            interaction_input = torch.cat([x, time_emb], dim=-1)
        else:
            interaction_input = x
        
        # Interaction stream
        interaction_output = self.interaction_stream(interaction_input)
        
        # Semantic stream
        if user_semantic is not None:
            semantic_output = self.semantic_stream(user_semantic)
        else:
            # If no semantic input, use zero vector
            batch_size = x.shape[0]
            semantic_output = torch.zeros(batch_size, 
                                         self.interaction_stream[-1].out_features).to(x.device)
        
        # Merge the outputs of the two streams
        fused = torch.cat([interaction_output, semantic_output], dim=-1)
        
        # Fusion layer
        output = self.fusion_layers(fused)
        
        return output

class FiLMSemanticDNN(nn.Module):
    """
    Semantic diffusion neural network based on FiLM (Feature-wise Linear Modulation).
    Uses user semantics $s_u$ as a condition control signal to perform affine transformation on the hidden features of the denoising backbone network.
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

        # Timestep embedding layer
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 1. Define the dimensions of the Denoising Backbone network
        # Input no longer contains semantic vectors, only concatenates interaction vector x_t and timestep embedding tau(t)
        if self.time_type == "cat":
            backbone_input_dim = self.in_dims[0] + self.time_emb_dim
            in_dims_temp = [backbone_input_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        
        out_dims_temp = self.out_dims
        
        # Backbone network - input layers
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        # Backbone network - output layers
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        
        # 2. Define Semantic Controller
        # Target is the first hidden layer of the backbone network (i.e. in_dims_temp[1])
        self.film_target_dim = in_dims_temp[1] 
        
        if self.use_semantic:
            # Output dimension is 2 * target_dim, used to split into gamma (scaling) and beta (shifting)
            self.semantic_controller = nn.Sequential(
                nn.Linear(self.semantic_dim, semantic_hidden_dim),
                nn.ReLU(),
                nn.Linear(semantic_hidden_dim, 2 * self.film_target_dim)
            )
        else:
            self.semantic_controller = None
            
        # Execute weight initialization (including core Zero-Initialization)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights, strictly following SemanticDNN style and anti-collapse constraints of FiLM"""
        # --- Xavier initialization of backbone network and timestep embedding ---
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

        # --- Special initialization of Semantic Controller ---
        if self.use_semantic and self.semantic_controller is not None:
            # Previous feature extraction layers use regular Xavier initialization
            for layer in self.semantic_controller[:-1]:
                if isinstance(layer, nn.Linear):
                    size = layer.weight.size()
                    fan_out = size[0]
                    fan_in = size[1]
                    std = np.sqrt(2.0 / (fan_in + fan_out))
                    layer.weight.data.normal_(0.0, std)
                    if layer.bias is not None:
                        layer.bias.data.normal_(0.0, 0.001)
            
            # Extremely important: the weight and bias of the last layer must be initialized to 0
            # Ensure initial output gamma=0, beta=0, making the FiLM layer equivalent to Identity Mapping
            final_layer = self.semantic_controller[-1]
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)
    
    def forward(self, x, timesteps, user_semantic=None, item_embeddings=None):
        # 1. Timestep embedding
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        
        # 2. Construct backbone network input (only concatenates x_t and timestep embedding)
        if self.time_type == "cat":
            h = torch.cat([x, time_emb], dim=-1)
        else:
            h = x
            
        # 3. Forward pass: pass through the first hidden layer and activate
        h = self.in_layers[0](h)
        h = torch.tanh(h)
        
        # 4. ======= FiLM Semantic Modulation =======
        if self.use_semantic and user_semantic is not None:
            # Calculate modulation parameters
            film_params = self.semantic_controller(user_semantic) # [batch_size, 2 * film_target_dim]
            
            # Split into scaling factor gamma and shifting factor beta
            gamma, beta = torch.chunk(film_params, 2, dim=-1) # Both [batch_size, film_target_dim]
            
            # Execute modulation: h_new = (1 + gamma) * h + beta
            h = (1.0 + gamma) * h + beta

        # 5. Continue remaining forward pass of backbone network
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
    Diffusion neural network based on FiLM with output layer using "semantic dot product" (specifically designed for cold start).
    No longer relies on ID mapping, but rather predicts the user's "ideal item semantics" and performs a dot product score with the global item semantics library.
    """
    def __init__(self, in_dims, out_dims, emb_size, semantic_dim=1024,
                 time_type="cat", norm=False, dropout=0.5,
                 semantic_hidden_dim=256, use_semantic=True):
        super(FiLMDotProductDNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.n_items = out_dims[-1]  # The last dimension is the total number of items
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.semantic_dim = semantic_dim
        self.use_semantic = use_semantic
        self.norm = norm

        # Timestep embedding layer
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # 1. Define Denoising Backbone network dimensions (exclude the final n_items)
        if self.time_type == "cat":
            backbone_input_dim = self.in_dims[0] + self.time_emb_dim
            in_dims_temp = [backbone_input_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)

        # Here out_dims_temp discards the final n_items, retaining only the hidden layer dimension
        out_dims_temp = self.out_dims[:-1]

        # Backbone network - input layers
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])

        # Backbone network - output layers (only to the last hidden layer, e.g. 400)
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)

        # 2. Define FiLM Semantic Controller
        self.film_target_dim = in_dims_temp[1]
        if self.use_semantic:
            self.semantic_controller = nn.Sequential(
                nn.Linear(self.semantic_dim, semantic_hidden_dim),
                nn.ReLU(),
                nn.Linear(semantic_hidden_dim, 2 * self.film_target_dim)
            )
        else:
            self.semantic_controller = None

        # 3. NEW: Semantic Dot Product Output Layer
        # Map the final hidden layer back to the semantic space of dimension 1024
        last_hidden_dim = out_dims_temp[-1] if len(out_dims_temp) > 0 else in_dims_temp[-1]
        self.semantic_output_layer = nn.Linear(last_hidden_dim, self.semantic_dim)

        # To stabilize training, add learnable temperature scaling and item bias
        #self.tau = nn.Parameter(torch.ones(1) * 10.0) # Initial scaling by 10 times to prevent extremely small L2 dot product values
        #self.item_bias = nn.Parameter(torch.zeros(self.n_items))

        self.init_weights()

    def init_weights(self):
        """Initialize weights, including FiLM zero-initialization and regular Xavier"""
        layers_to_init = []
        if hasattr(self, 'in_layers'): layers_to_init.extend(self.in_layers)
        if hasattr(self, 'out_layers'): layers_to_init.extend(self.out_layers)
        layers_to_init.extend([self.emb_layer, self.semantic_output_layer])

        for layer in layers_to_init:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, 0, 0.001)

        # FiLM Controller zero-initialization to prevent collapse
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
        Added item_embeddings parameter [n_items, semantic_dim]
        """
        # 1. Timestep embedding and input processing
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        if self.norm: x = F.normalize(x)
        x = self.drop(x)

        h = torch.cat([x, time_emb], dim=-1) if self.time_type == "cat" else x

        # 2. First layer and FiLM Modulation
        h = self.in_layers[0](h)
        h = torch.tanh(h)
        if self.use_semantic and user_semantic is not None:
            film_params = self.semantic_controller(user_semantic)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            h = (1.0 + gamma) * h + beta

        # 3. Backbone network forward pass
        for i in range(1, len(self.in_layers)):
            h = self.in_layers[i](h)
            h = torch.tanh(h)
        for layer in self.out_layers:
            h = layer(h)
            h = torch.tanh(h)

        # 4. Semantic Prediction and Dot Product Scoring
        # Predict the user's ideal item semantics at this moment [batch_size, 1024]
        pred_semantic = self.semantic_output_layer(h)

        if item_embeddings is None:
            raise ValueError("item_embeddings MUST be provided for FiLMDotProductDNN")

        # L2 Normalization (prevent dimension explosion)
        pred_semantic_norm = F.normalize(pred_semantic, p=2, dim=-1)
        item_embeddings_norm = F.normalize(item_embeddings, p=2, dim=-1)

        # Matrix dot product: [batch_size, 1024] x [1024, n_items] -> [batch_size, n_items]
        logits = torch.matmul(pred_semantic_norm, item_embeddings.transpose(0, 1))

        # Apply fixed temperature parameter (tau=0.1) and Diffusion scaling factor (scale=10.0)
        # Refers to SimCLR standard setting, magnifying differences while ensuring the overall numerical range conforms to Diffusion Loss demands
        tau = 0.1
        scale = 10.0
        out = (logits / tau) * scale

        return out