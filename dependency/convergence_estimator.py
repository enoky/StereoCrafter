import torch
import torch.nn.functional as F
import os
import logging
from .u2net import U2NETP

logger = logging.getLogger(__name__)

class ConvergenceEstimator:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConvergenceEstimator, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path=None, device=None):
        if hasattr(self, 'model'):
            return
            
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine model path if not provided
        if model_path is None:
            # Assumes the model is in the same directory as this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "iw3_sod_v1_20260122.pth")
            
        if not os.path.exists(model_path):
            logger.warning(f"Convergence model not found at {model_path}")
            self.model = None
            return

        try:
            # Initialize U2NETP with 6 input channels
            self.model = U2NETP(in_ch=6, out_ch=1)
            
            # Load weights
            logger.info(f"Loading convergence model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract state_dict if it's wrapped
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Strip 'u2netp.' prefix if present (artifact of how it was saved)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("u2netp."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device).eval()
            logger.info("Convergence model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load convergence model: {e}")
            self.model = None

    def preprocess(self, rgb_tensor, depth_tensor):
        """
        Prepares input for U2NETP (6-channel).
        rgb_tensor: [B, 3, H, W]
        depth_tensor: [B, 1, H, W] (normalized 0-1)
        """
        # Resize to 192x192
        rgb_small = F.interpolate(rgb_tensor, (192, 192), mode="bilinear", align_corners=False)
        depth_small = F.interpolate(depth_tensor, (192, 192), mode="bilinear", align_corners=False)
        
        # Feature Engineering: Depth, Sqrt(Depth), Square(Depth)
        # Ensure depth is clamped 0-1, though it should be already
        depth_small = torch.clamp(depth_small, 0.0, 1.0)
        
        depth_sqrt = torch.pow(depth_small, 0.5)
        depth_pow = torch.pow(depth_small, 2.0)
        
        # Concatenate: RGB (3) + Depth (1) + Sqrt (1) + Square (1) = 6 channels
        x = torch.cat([rgb_small, depth_small, depth_sqrt, depth_pow], dim=1)
        return x, depth_small

    def calculate_robust_depth(self, saliency_map, depth_small, user_convergence_ratio=0.5):
        """
        Estimates convergence point from saliency map and depth.
        """
        batch_size = depth_small.shape[0]
        results = []

        for i in range(batch_size):
            # Threshold saliency
            mask = (saliency_map[i, 0] > 0.5)
            d_vals = depth_small[i, 0][mask]

            if d_vals.numel() == 0:
                # Fallback if no saliency detected
                results.append(0.5)
                continue

            # Robust statistics (10th and 90th percentile)
            q01 = torch.quantile(d_vals, 0.1)
            q09 = torch.quantile(d_vals, 0.9)

            center = (q01 + q09) / 2.0
            obj_range = q09 - q01

            if obj_range < 1e-6:
                q_pos = q01
            else:
                # Expanded range logic from original estimator
                expanded_range = obj_range * 3.0
                q_pos = center + (user_convergence_ratio - 0.5) * expanded_range

            q_pos = torch.clamp(q_pos, 0.0, 1.0)
            results.append(q_pos.item())

        return results

    @torch.inference_mode()
    def predict(self, rgb_tensor, depth_tensor, user_ratio=0.5):
        """
        Main prediction method.
        rgb_tensor: [B, 3, H, W] (float 0-1)
        depth_tensor: [B, 1, H, W] (float 0-1)
        """
        if self.model is None:
            return [0.5] * rgb_tensor.shape[0]

        rgb_tensor = rgb_tensor.to(self.device)
        depth_tensor = depth_tensor.to(self.device)

        input_tensor, depth_small = self.preprocess(rgb_tensor, depth_tensor)
        
        # Inference
        outputs = self.model(input_tensor)
        # U2NETP returns tuple of outputs, we want d0 (the first one)
        saliency = outputs[0]
        
        # Ensure range [0, 1] (sigmoid is already applied in model forward but good to be safe if model changes)
        # The provided model code applies sigmoid at the end, so outputs[0] is in [0, 1].
        
        return self.calculate_robust_depth(saliency, depth_small, user_ratio)
