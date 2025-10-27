"""
BrownianTree Noise Sampler for SDE schedulers
Provides temporally correlated noise for better video quality

Adapted from:
- sd-webui-forge-classic/modules_forge/packages/k_diffusion/sampling.py
- torchsde BrownianTree implementation
"""

import torch
try:
    import torchsde
    TORCHSDE_AVAILABLE = True
except ImportError:
    TORCHSDE_AVAILABLE = False
    print("Warning: torchsde not available. BrownianTreeNoiseSampler will fall back to white noise.")


class BatchedBrownianTree:
    """
    A wrapper around torchsde.BrownianTree that enables batches of entropy.
    
    Provides temporally correlated noise samples for SDE schedulers.
    Each batch item can have its own seed for reproducible but varied results.
    """

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        """
        Initialize batched Brownian tree.
        
        Args:
            x: Tensor with shape to match (batch_size, ...)
            t0: Start time
            t1: End time
            seed: Random seed (int or list of ints for batch)
            **kwargs: Additional args passed to torchsde.BrownianTree
        """
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        
        if TORCHSDE_AVAILABLE:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]
        else:
            # Fallback: store seed for white noise generation
            self.trees = None
            self.seeds = seed
            self.shape = x.shape
            self.device = x.device
            self.dtype = x.dtype

    @staticmethod
    def sort(a, b):
        """Sort times and track direction."""
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        """
        Sample correlated noise between t0 and t1.
        
        Args:
            t0: Start time for this step
            t1: End time for this step
            
        Returns:
            Correlated noise tensor
        """
        t0, t1, sign = self.sort(t0, t1)
        
        if TORCHSDE_AVAILABLE:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
            return w if self.batched else w[0]
        else:
            # Fallback to white noise
            if self.batched:
                return torch.randn(self.shape, device=self.device, dtype=self.dtype)
            else:
                return torch.randn(self.shape, device=self.device, dtype=self.dtype)[0]


class BrownianTreeNoiseSampler:
    """
    A noise sampler backed by a torchsde.BrownianTree.
    
    Provides temporally correlated noise for SDE schedulers, which produces
    smoother and more natural results compared to independent white noise.
    
    This is especially important for video generation where temporal coherence matters.
    
    Args:
        x (Tensor): The tensor whose shape, device and dtype to use
        sigma_min (float): The low end of the valid sigma interval
        sigma_max (float): The high end of the valid sigma interval
        seed (int or List[int]): Random seed. If a list, one seed per batch item
        transform (callable): Function that maps sigma to internal timestep
        
    Example:
        >>> latents = torch.randn(1, 4, 8, 64, 64)
        >>> noise_sampler = BrownianTreeNoiseSampler(latents, 0.001, 14.6, seed=42)
        >>> noise = noise_sampler(sigma_current, sigma_next)
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        """Initialize Brownian tree noise sampler."""
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        """
        Sample correlated noise for step from sigma to sigma_next.
        
        The noise is scaled by sqrt(|t1 - t0|) to have the correct variance.
        
        Args:
            sigma: Current sigma value
            sigma_next: Next sigma value
            
        Returns:
            Temporally correlated noise tensor
        """
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


def test_brownian_sampler():
    """Test that BrownianTreeNoiseSampler works correctly."""
    print("Testing BrownianTreeNoiseSampler...")
    
    # Create sample latents (video format: batch, channels, frames, height, width)
    x = torch.randn(1, 4, 8, 64, 64)
    
    # Create noise sampler
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min=0.001, sigma_max=14.6, seed=42)
    
    # Sample noise for two consecutive steps
    noise1 = noise_sampler(sigma=1.0, sigma_next=0.5)
    noise2 = noise_sampler(sigma=0.5, sigma_next=0.25)
    
    print(f"Noise 1 shape: {noise1.shape}")
    print(f"Noise 2 shape: {noise2.shape}")
    print(f"Noise 1 mean: {noise1.mean().item():.4f}, std: {noise1.std().item():.4f}")
    print(f"Noise 2 mean: {noise2.mean().item():.4f}, std: {noise2.std().item():.4f}")
    
    # Test reproducibility
    noise_sampler2 = BrownianTreeNoiseSampler(x, sigma_min=0.001, sigma_max=14.6, seed=42)
    noise1_repeat = noise_sampler2(sigma=1.0, sigma_next=0.5)
    
    if TORCHSDE_AVAILABLE:
        diff = (noise1 - noise1_repeat).abs().max().item()
        print(f"Reproducibility test - max difference: {diff:.2e} (should be ~0)")
        assert diff < 1e-5, "BrownianTree should be reproducible with same seed!"
    else:
        print("torchsde not available - using white noise fallback")
    
    print("✓ BrownianTreeNoiseSampler test passed!")


if __name__ == "__main__":
    test_brownian_sampler()

