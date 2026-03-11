import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class WaveletBlock(nn.Module):
    def __init__(self, seq_len, wavelet='db4', level=2, mode='zero', hidden_dim=64):
        """
        Advanced Wavelet Transform Block for time series data with adaptive pooling
        
        Args:
            seq_len (int): Input sequence length
            wavelet (str): Wavelet to use (default: 'db4')
            level (int): Decomposition level (default: 2)
            mode (str): Padding mode (default: 'zero')
            hidden_dim (int): Hidden dimension for attention (default: 64)
        """
        super(WaveletBlock, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.seq_len = seq_len
        self.level = level
        self.hidden_dim = hidden_dim
        
        # Use adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(hidden_dim)
        
        # Attention with fixed dimensions
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Project back to original sequence length
        self.projection = nn.Linear(hidden_dim, seq_len)
        
    def dwt_forward(self, x):
        """
        Forward wavelet transform
        
        Args:
            x (torch.Tensor): Input tensor [B, L]
            
        Returns:
            tensor of flattened coeffs
        """
        device = x.device
        batch_size = x.shape[0]
        x_np = x.detach().cpu().numpy()
        
        # Store coefficient lists for each sample
        all_coeffs = []
        all_shapes = []
        all_flattened = []
        
        for i in range(batch_size):
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(x_np[i], self.wavelet, mode=self.mode, level=self.level)
            all_coeffs.append(coeffs)
            
            # Record shapes for reconstruction
            shapes = [c.shape[0] for c in coeffs]
            all_shapes.append(shapes)
            
            # Flatten all coefficients into a single array
            flattened = np.concatenate([c for c in coeffs])
            all_flattened.append(flattened)
        
        # Convert to tensors
        coeffs_tensor = torch.tensor(np.stack(all_flattened), dtype=x.dtype, device=device)
        
        return coeffs_tensor, all_shapes
    
    def iwt_forward(self, x_orig, coeffs_tensor, shapes):
        """
        Inverse wavelet transform
        
        Args:
            x_orig (torch.Tensor): Original input for shape reference
            coeffs_tensor (torch.Tensor): Processed coefficients 
            shapes (list): List of shapes for each level
            
        Returns:
            torch.Tensor: Reconstructed signal
        """
        device = coeffs_tensor.device
        batch_size = coeffs_tensor.shape[0]
        coeffs_np = coeffs_tensor.detach().cpu().numpy()
        
        # Project back to sequence length if we're using a different dimension
        if coeffs_tensor.shape[1] != self.seq_len:
            # We'll use the original input directly instead of reconstruction
            return x_orig
        
        reconstructed = []
        
        for i in range(batch_size):
            # Try to reconstruct if we can, otherwise use the original
            try:
                # Split flattened coefficients back to original shapes
                coeffs_list = []
                idx = 0
                for shape in shapes[i]:
                    if idx + shape <= len(coeffs_np[i]):
                        coeffs_list.append(coeffs_np[i][idx:idx+shape])
                        idx += shape
                
                # Perform inverse transform
                rec = pywt.waverec(coeffs_list, self.wavelet, mode=self.mode)
                
                # Trim to original length if needed
                if len(rec) > self.seq_len:
                    rec = rec[:self.seq_len]
                elif len(rec) < self.seq_len:
                    # Pad if necessary
                    pad = np.zeros(self.seq_len - len(rec))
                    rec = np.concatenate([rec, pad])
                    
                reconstructed.append(rec)
            except:
                # If reconstruction fails, use original input
                reconstructed.append(x_orig[i].cpu().numpy())
            
        # Convert back to tensor
        reconstructed_tensor = torch.tensor(np.stack(reconstructed), dtype=x_orig.dtype, device=device)
        return reconstructed_tensor
    
    def forward(self, x):
        """
        Process using wavelet transform with attention mechanism
        
        Args:
            x (torch.Tensor): Input tensor [B, L]
            
        Returns:
            torch.Tensor: Processed tensor
        """
        # Store original for fallback
        x_orig = x.clone()
        
        # Apply forward DWT
        coeffs_tensor, shapes = self.dwt_forward(x)
        
        # Use adaptive pooling to get fixed dimension
        # Need to unsqueeze to add channel dimension for 1D pooling
        coeffs_tensor_reshaped = coeffs_tensor.unsqueeze(1)
        
        # Check for MPS device and move pooling to CPU if necessary
        device = x.device  # Get the device of the input tensor
        if device.type == 'mps':
            pooled_coeffs = self.adaptive_pool(coeffs_tensor_reshaped.cpu()).squeeze(1)
            pooled_coeffs = pooled_coeffs.to(device)
        else:
            pooled_coeffs = self.adaptive_pool(coeffs_tensor_reshaped).squeeze(1)
        
        # Apply attention to pooled coefficients
        attention_weights = self.attention(pooled_coeffs)
        modified_coeffs = pooled_coeffs * attention_weights
        
        # Project back to original sequence length
        output = self.projection(modified_coeffs)
        
        # Combine with original input as residual connection
        # This ensures we maintain the original signal information
        return x + output * 0.1  # Scale factor to prevent instability