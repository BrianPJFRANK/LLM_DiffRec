# models/semantic_diffusion.py
import torch as th
import torch.nn as nn
from models.gaussian_diffusion import GaussianDiffusion, ModelMeanType

class SemanticGaussianDiffusion(GaussianDiffusion):
    """
    支持語義輸入的高斯擴散類
    擴展原始GaussianDiffusion以支持語義條件
    """
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,
                 steps, device, history_num_per_term=10, beta_fixed=True):
        super().__init__(mean_type, noise_schedule, noise_scale, noise_min, noise_max,
                        steps, device, history_num_per_term, beta_fixed)
    
    def training_losses(self, model, x_start, user_semantic=None, reweight=False):
        """
        計算訓練損失，支持語義輸入
        
        Args:
            model: 預測模型
            x_start: 原始交互向量 [batch_size, n_items]
            user_semantic: 用戶語義向量 [batch_size, semantic_dim] 或 None
            reweight: 是否使用重要性採樣
        
        Returns:
            terms: 包含損失的字典
        """
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)
        
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start
        
        terms = {}
        
        # 修改：傳遞語義向量給模型
        if user_semantic is not None:
            model_output = model(x_t, ts, user_semantic)
        else:
            model_output = model(x_t, ts)
        
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]
        
        assert model_output.shape == target.shape == x_start.shape
        
        mse = self.mean_flat((target - model_output) ** 2)
        
        # 修復：確保 loss 變量總是被初始化
        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = self.mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)
            loss = mse  # 當不使用reweight時，直接使用MSE
        
        terms["loss"] = weight * loss
        
        # 更新 Lt_history & Lt_count (保持不變)
        for t, loss_value in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss_value.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss_value.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss_value)
                    raise ValueError
        
        terms["loss"] /= pt
        return terms
    
    def p_sample(self, model, x_start, steps, user_semantic=None, sampling_noise=False):
        """
        採樣過程，支持語義輸入
        
        Args:
            model: 預測模型
            x_start: 起始向量
            steps: 採樣步數
            user_semantic: 用戶語義向量
            sampling_noise: 是否添加採樣噪聲
        
        Returns:
            x_t: 採樣結果
        """
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)
        
        indices = list(range(self.steps))[::-1]
        
        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                if user_semantic is not None:
                    x_t = model(x_t, t, user_semantic)
                else:
                    x_t = model(x_t, t)
            return x_t
        
        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            # 修改：傳遞語義向量給p_mean_variance
            out = self.p_mean_variance(model, x_t, t, user_semantic)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t
    
    def p_mean_variance(self, model, x, t, user_semantic=None):
        """
        計算均值方差，支持語義輸入
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        # 修改：傳遞語義向量給模型
        if user_semantic is not None:
            model_output = model(x, t, user_semantic)
        else:
            model_output = model(x, t)
        
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        
        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    @staticmethod
    def mean_flat(tensor):
        """輔助函數：計算非批次維度的平均值"""
        return tensor.mean(dim=list(range(1, len(tensor.shape))))