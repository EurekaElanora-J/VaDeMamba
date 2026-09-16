import numpy as np
import torch
import scipy.ndimage as ndimage
from scipy.fft import dctn

def compute_gradient_complexity(img_region):
    """
    Computes 3D gradient magnitude complexity across all available channels.
    img_region: (C, S, S, S) numpy array or torch tensor.
    Returns: scalar float.
    """
    if isinstance(img_region, torch.Tensor):
        img_np = img_region.cpu().numpy()
    else:
        img_np = img_region
        
    c_grads = []
    for c in range(img_np.shape[0]):
        channel_data = img_np[c]
        # 3D spatial gradients along axes 0, 1, 2
        gz, gy, gx = np.gradient(channel_data)
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        c_grads.append(np.mean(grad_mag))
        
    return float(np.mean(c_grads))

def compute_frequency_complexity(img_region, cutoff_ratio=0.5):
    """
    Computes ratio of high-frequency energy to total energy using 3D DCT.
    img_region: (C, S, S, S) numpy array or torch tensor.
    cutoff_ratio: fraction of Nyquist frequency defining 'high frequency' (default 0.5).
    Returns: scalar float in [0, 1].
    """
    if isinstance(img_region, torch.Tensor):
        img_np = img_region.cpu().numpy()
    else:
        img_np = img_region
        
    S = img_region.shape[-1]
    # Build radial frequency distance grid
    z, y, x = np.ogrid[:S, :S, :S]
    dist = np.sqrt(z**2 + y**2 + x**2)
    cutoff = cutoff_ratio * np.sqrt(3) * S / 2.0
    high_freq_mask = dist > cutoff
    
    hf_ratios = []
    for c in range(img_np.shape[0]):
        channel_data = img_np[c]
        # 3D type-2 DCT with ortho norm
        dct_coeffs = dctn(channel_data, norm='ortho')
        energy = dct_coeffs ** 2
        total_energy = np.sum(energy)
        
        if total_energy < 1e-8:
            hf_ratios.append(0.0)
        else:
            hf_energy = np.sum(energy[high_freq_mask])
            hf_ratios.append(float(hf_energy / total_energy))
            
    return float(np.mean(hf_ratios))

def compute_intensity_entropy(img_region, num_bins=32):
    """
    Computes local Shannon entropy of voxel intensities.
    img_region: (C, S, S, S) numpy array or torch tensor.
    Returns: scalar float.
    """
    if isinstance(img_region, torch.Tensor):
        img_np = img_region.cpu().numpy()
    else:
        img_np = img_region
        
    c_entropies = []
    for c in range(img_np.shape[0]):
        channel_data = img_np[c].flatten()
        # Non-zero or all voxels
        hist, _ = np.histogram(channel_data, bins=num_bins, density=True)
        probs = hist / (np.sum(hist) + 1e-12)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        c_entropies.append(entropy)
        
    return float(np.mean(c_entropies))

def compute_model_uncertainty(prob_region):
    """
    Computes mean Shannon predictive entropy over 4 classes in region.
    prob_region: (4, S, S, S) torch tensor or numpy array of probabilities.
    Returns: scalar float.
    """
    if isinstance(prob_region, np.ndarray):
        prob = torch.from_numpy(prob_region)
    else:
        prob = prob_region
        
    eps = 1e-7
    # H = - sum_c p_c * log(p_c)
    entropy_map = -torch.sum(prob * torch.log(prob + eps), dim=0) # (S, S, S)
    return float(entropy_map.mean().item())

def extract_all_complexity_signals(img_region, prob1_region=None):
    """
    Extracts all 4 candidate complexity predictors for a given 3D region.
    """
    c_grad = compute_gradient_complexity(img_region)
    c_freq = compute_frequency_complexity(img_region)
    c_entropy = compute_intensity_entropy(img_region)
    
    if prob1_region is not None:
        c_uncertainty = compute_model_uncertainty(prob1_region)
    else:
        c_uncertainty = None
        
    return {
        "C_grad": c_grad,
        "C_freq": c_freq,
        "C_entropy": c_entropy,
        "C_uncertainty": c_uncertainty
    }

if __name__ == "__main__":
    # Test on synthetic 16^3 and 8^3 blocks
    synthetic_smooth = torch.ones(4, 16, 16, 16) * 0.5
    synthetic_noisy = torch.randn(4, 16, 16, 16)
    synthetic_prob = torch.softmax(torch.randn(4, 16, 16, 16), dim=0)
    
    res_smooth = extract_all_complexity_signals(synthetic_smooth, synthetic_prob)
    res_noisy = extract_all_complexity_signals(synthetic_noisy, synthetic_prob)
    
    print("Smooth Region Complexity:", res_smooth)
    print("Noisy Region Complexity:", res_noisy)
    assert res_noisy["C_grad"] > res_smooth["C_grad"], "Gradient complexity test failed"
    assert res_noisy["C_freq"] > res_smooth["C_freq"], "Frequency complexity test failed"
    print("Complexity estimators verified successfully!")
