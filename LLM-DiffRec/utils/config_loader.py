# config_loader.py
"""
Configuration loader for semantic diffusion model
"""

import yaml
import argparse
import os

def load_config(config_path="config_semantic.yaml", override_args=None):
    """
    加載配置文件並與命令行參數合併
    
    Args:
        config_path: 配置文件路徑
        override_args: 命令行參數（可選）
    
    Returns:
        config: 合併後的配置字典
    """
    # 默認配置
    default_config = {
        'dataset': {
            'name': 'amazon-instruments',
            'data_path': './datasets/',
            'use_cold_start': False
        },
        'semantic': {
            'enabled': True,
            'model_type': 'semantic',
            'semantic_dim': 768,
            'semantic_proj_dim': 128,
            'aggregation_method': 'mean'
        },
        'model': {
            'time_type': 'cat',
            'dims': [1000],
            'emb_size': 10,
            'norm': False,
            'dropout': 0.5
        },
        'diffusion': {
            'mean_type': 'x0',
            'steps': 5,
            'noise_schedule': 'linear-var',
            'noise_scale': 0.0001,
            'noise_min': 0.0005,
            'noise_max': 0.005,
            'reweight': True
        },
        'training': {
            'lr': 0.0001,
            'weight_decay': 0.0,
            'batch_size': 400,
            'epochs': 1000,
            'early_stop_patience': 20,
            'save_path': './saved_models_semantic/'
        },
        'inference': {
            'sampling_steps': 0,
            'sampling_noise': False,
            'topN': [10, 20, 50, 100],
            'tst_w_val': False
        },
        'device': {
            'cuda': True,
            'gpu': '0'
        },
        'experiment': {
            'log_name': 'semantic_experiment',
            'round': 1,
            'save_logs': True,
            'log_dir': './logs/'
        },
        'cold_start': {
            'enabled': False,
            'cold_start_ratio': 0.3,
            'analyze_similarity': True
        }
    }
    
    # 加載配置文件（如果存在）
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # 深度合併配置
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        config = deep_merge(default_config, file_config)
    else:
        print(f"Config file {config_path} not found, using defaults")
        config = default_config
    
    # 用命令行參數覆蓋（如果提供）
    if override_args is not None:
        args_dict = vars(override_args)
        
        # 映射命令行參數到配置鍵
        arg_mapping = {
            'dataset': 'dataset.name',
            'data_path': 'dataset.data_path',
            'use_semantic': 'semantic.enabled',
            'model_type': 'semantic.model_type',
            'semantic_dim': 'semantic.semantic_dim',
            'semantic_proj_dim': 'semantic.semantic_proj_dim',
            'lr': 'training.lr',
            'weight_decay': 'training.weight_decay',
            'batch_size': 'training.batch_size',
            'epochs': 'training.epochs',
            'dims': 'model.dims',
            'emb_size': 'model.emb_size',
            'mean_type': 'diffusion.mean_type',
            'steps': 'diffusion.steps',
            'noise_scale': 'diffusion.noise_scale',
            'noise_min': 'diffusion.noise_min',
            'noise_max': 'diffusion.noise_max',
            'sampling_steps': 'inference.sampling_steps',
            'reweight': 'diffusion.reweight',
            'cuda': 'device.cuda',
            'gpu': 'device.gpu',
            'log_name': 'experiment.log_name',
            'round': 'experiment.round',
            'cold_start': 'cold_start.enabled'
        }
        
        for arg_key, config_path in arg_mapping.items():
            if arg_key in args_dict and args_dict[arg_key] is not None:
                # 設置配置值
                path_parts = config_path.split('.')
                current = config
                for part in path_parts[:-1]:
                    current = current[part]
                current[path_parts[-1]] = args_dict[arg_key]
    
    return config

def save_config(config, path):
    """保存配置到文件"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def print_config(config, indent=0):
    """打印配置"""
    for key, value in config.items():
        if isinstance(value, dict):
            print(' ' * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(' ' * indent + f"{key}: {value}")

if __name__ == "__main__":
    # 測試配置加載
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_semantic.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    print("Loaded configuration:")
    print_config(config)