"""
Super-resolution module for upscaling rendered images.

Supports Real-ESRGAN models for photorealistic upscaling.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


@dataclass
class SuperResolutionConfig:
    """Configuration for super-resolution upscaling."""

    model_name: str = "RealESRGAN_x4plus"
    """Model to use: RealESRGAN_x4plus, RealESRNet_x4plus, or RealESRGAN_x4plus_anime_6B"""

    scale: int = 4
    """Upscaling factor (2 or 4)"""

    model_path: Optional[Path] = None
    """Path to model weights. If None, uses default: data/pretrained/sr/{model_name}.pth"""

    device: str = "cuda"
    """Device to run on: cuda or cpu"""

    fp16: bool = True
    """Use FP16 for faster inference (requires GPU)"""

    tile_size: int = 0
    """Tile size for large images (0 = no tiling). Use 512 for 4K+ images"""

    tile_pad: int = 10
    """Padding for tiles to avoid seams"""

    pre_pad: int = 0
    """Pre-padding before inference"""


class SuperResolution:
    """
    Super-resolution upscaler using Real-ESRGAN.

    Usage:
        # 1. Download models first
        python scripts/download_sr_models.py

        # 2. Create upscaler
        sr = SuperResolution(SuperResolutionConfig(model_name="RealESRGAN_x4plus"))

        # 3. Upscale
        img_hr = sr.upscale(img_lr)

    Example:
        from mugs.sensors import GaussianSensor
        from mugs.postprocess import SuperResolution, SuperResolutionConfig

        # Low-res rendering (fast)
        sensor = GaussianSensor(GaussianSensorConfig(width=320, height=240))
        img_lr = sensor.render(model, data, camera_name)

        # Optional high-res upscaling
        sr = SuperResolution(SuperResolutionConfig())
        img_hr = sr.upscale(img_lr)  # 1280×960
    """

    def __init__(self, config: SuperResolutionConfig):
        self.config = config
        self._upsampler = None
        self._model_loaded = False

        # Lazy loading - only load when first used
        # This allows importing the module without having models installed

    def _load_model(self):
        """Lazy load the Real-ESRGAN model."""
        if self._model_loaded:
            return

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as e:
            raise ImportError(
                "Real-ESRGAN not installed. Install with:\n"
                "  pip install realesrgan basicsr\n"
                "Or install all dependencies:\n"
                "  pip install -r requirements.txt"
            ) from e

        # Determine model path
        if self.config.model_path is None:
            default_path = Path("data/pretrained/sr") / f"{self.config.model_name}.pth"
            self.config.model_path = default_path

        # Check if model exists
        if not self.config.model_path.exists():
            model_dir = self.config.model_path.parent
            raise FileNotFoundError(
                f"Model not found: {self.config.model_path}\n\n"
                f"Download pretrained models first:\n"
                f"  python scripts/download_sr_models.py\n\n"
                f"Or manually download to {model_dir}/:\n"
                f"  - RealESRGAN_x4plus.pth (general purpose, 64MB)\n"
                f"    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth\n"
                f"  - RealESRNet_x4plus.pth (faster, 64MB)\n"
                f"    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth\n"
                f"  - RealESRGAN_x4plus_anime_6B.pth (anime style, 17MB)\n"
                f"    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth\n"
            )

        # Create model architecture
        if "anime" in self.config.model_name.lower():
            # Anime model uses different architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=self.config.scale,
            )
        else:
            # Standard Real-ESRGAN architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=self.config.scale,
            )

        # Create upsampler
        try:
            self._upsampler = RealESRGANer(
                scale=self.config.scale,
                model_path=str(self.config.model_path),
                model=model,
                tile=self.config.tile_size,
                tile_pad=self.config.tile_pad,
                pre_pad=self.config.pre_pad,
                half=self.config.fp16 and self.config.device == "cuda",
                device=self.config.device,
            )
            self._model_loaded = True
            print(f"✅ Loaded SR model: {self.config.model_name} ({self.config.scale}x)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Real-ESRGAN model from {self.config.model_path}\n"
                f"Error: {e}\n\n"
                f"Try re-downloading the model:\n"
                f"  python scripts/download_sr_models.py --model {self.config.model_name}"
            ) from e

    def upscale(self, img: np.ndarray, outscale: Optional[int] = None) -> np.ndarray:
        """
        Upscale a single image.

        Args:
            img: Input image (H, W, 3) in RGB format, uint8 or float32
            outscale: Output scale (default: use config.scale)

        Returns:
            Upscaled image (H*scale, W*scale, 3) in uint8 RGB format

        Example:
            img_lr = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            img_hr = sr.upscale(img_lr)  # (960, 1280, 3)
        """
        # Lazy load model on first use
        if not self._model_loaded:
            self._load_model()

        # Convert float to uint8 if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Upscale
        try:
            output, _ = self._upsampler.enhance(
                img, outscale=outscale if outscale is not None else self.config.scale
            )
            return output
        except Exception as e:
            raise RuntimeError(
                f"Super-resolution failed: {e}\n"
                f"Input shape: {img.shape}, dtype: {img.dtype}\n"
                f"Model: {self.config.model_name}, scale: {self.config.scale}"
            ) from e

    def batch_upscale(
        self, imgs: List[np.ndarray], outscale: Optional[int] = None, show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Upscale a batch of images.

        Args:
            imgs: List of input images
            outscale: Output scale (default: use config.scale)
            show_progress: Show progress bar

        Returns:
            List of upscaled images

        Example:
            imgs_lr = [sensor.render(...) for _ in range(100)]
            imgs_hr = sr.batch_upscale(imgs_lr)
        """
        outputs = []

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(imgs, desc="Upscaling images")
            except ImportError:
                print(f"Upscaling {len(imgs)} images...")
                iterator = imgs
        else:
            iterator = imgs

        for img in iterator:
            outputs.append(self.upscale(img, outscale))

        return outputs

    @staticmethod
    def available_models() -> List[str]:
        """List available pretrained models."""
        return [
            "RealESRGAN_x4plus",  # Best quality, general purpose
            "RealESRNet_x4plus",  # Faster, slightly lower quality
            "RealESRGAN_x4plus_anime_6B",  # Optimized for anime/cartoon
        ]

    def __repr__(self) -> str:
        return (
            f"SuperResolution(\n"
            f"  model={self.config.model_name},\n"
            f"  scale={self.config.scale}x,\n"
            f"  device={self.config.device},\n"
            f"  loaded={self._model_loaded}\n"
            f")"
        )
