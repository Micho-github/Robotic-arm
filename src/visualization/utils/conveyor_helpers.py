import numpy as np


def _chw_to_hwc(image_chw):
    """Convert image from CHW (Channels, Height, Width) to HWC (Height, Width, Channels) format."""
    if isinstance(image_chw, np.ndarray) and image_chw.ndim == 3 and image_chw.shape[0] == 3:
        return np.transpose(image_chw, (1, 2, 0))
    return image_chw


def _resize_hwc(image_hwc, size=48):
    """Resize an HWC format image to the specified size."""
    try:
        from PIL import Image
        img8 = (np.clip(image_hwc, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(img8, mode="RGB")
        pil = pil.resize((size, size), resample=Image.BILINEAR)
        return (np.asarray(pil).astype(np.float32) / 255.0)
    except:
        h, w = image_hwc.shape[:2]
        ys = (np.linspace(0, h - 1, size)).astype(int)
        xs = (np.linspace(0, w - 1, size)).astype(int)
        return image_hwc[np.ix_(ys, xs)]


def _as_rgba(image_hwc, alpha=1.0):
    """Convert HWC image to RGBA format by adding alpha channel."""
    img = np.clip(image_hwc, 0, 1)
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    a = np.full((img.shape[0], img.shape[1], 1), float(alpha), dtype=np.float32)
    return np.concatenate([img.astype(np.float32), a], axis=2)


def _draw_image_decal_3d(ax, center_x, center_y, center_z, image_hwc, size_world=0.65, zorder=10):
    """Draw an image as a 3D surface decal in matplotlib 3D plot."""
    img = _as_rgba(image_hwc, alpha=0.98)
    h, w = img.shape[:2]
    xs = np.linspace(center_x - size_world / 2, center_x + size_world / 2, w)
    ys = np.linspace(center_y - size_world / 2, center_y + size_world / 2, h)
    X, Y = np.meshgrid(xs, ys)
    Z = np.ones_like(X) * float(center_z)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=img, shade=False, linewidth=0, zorder=zorder)
    return surf

