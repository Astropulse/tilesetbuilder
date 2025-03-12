import math
import numpy as np
from PIL import Image

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
import colour
from minisom import MiniSom




def mean_shift_quantize(img: Image.Image, quantile: float = 0.06) -> Image.Image:
    """
    Performs mean-shift quantization on the given image in the Oklab color space.
    This function uses scikit-learn's MeanShift and colour-science to convert
    between color spaces.

    Parameters
    ----------
    img : Image.Image
        Input PIL image.
    quantile : float, optional
        Quantile for estimating the bandwidth. Default is 0.06.

    Returns
    -------
    Image.Image
        The quantized PIL image.
    """
    # 0. Handle RGBA: threshold alpha channel if present
    has_alpha = (img.mode == "RGBA")
    mask = None
    if has_alpha:
        rgba_arr = np.array(img, dtype=np.uint8)
        alpha_channel = rgba_arr[:, :, 3]
        mask = (alpha_channel > 50).astype(np.uint8) * 255
        rgba_arr[:, :, 3] = mask
        thresholded_img = Image.fromarray(rgba_arr, mode="RGBA")

        white_bg = Image.new("RGBA", thresholded_img.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(white_bg, thresholded_img)
        img_for_quant = composited.convert("RGB")
    else:
        img_for_quant = img.convert("RGB")

    orig_w, orig_h = img_for_quant.size
    x = math.sqrt(orig_w * orig_h)
    if x < 1:
        return 1.0

    factor = (0.00001*x**2 + 0.07*x + 14.9) / x
    down_w = max(1, int(round(orig_w * factor)))
    down_h = max(1, int(round(orig_h * factor)))

    print(f"Original: {orig_w}x{orig_h}, factor={factor:.3f}, down to {down_w}x{down_h}")

    downscaled_img = img_for_quant.resize((down_w, down_h), Image.Resampling.NEAREST)
    downscaled_arr = np.array(downscaled_img, dtype=np.uint8)
    downscaled_float = downscaled_arr.astype(np.float64) / 255.0

    down_xyz = colour.sRGB_to_XYZ(downscaled_float)
    down_oklab = colour.XYZ_to_Oklab(down_xyz)
    pixels_down = down_oklab.reshape(-1, 3)

    bandwidth = estimate_bandwidth(pixels_down, quantile=quantile)
    ms = MeanShift(bin_seeding=True, bandwidth=bandwidth)
    ms.fit(pixels_down)

    centers_oklab = ms.cluster_centers_
    n_clusters = len(centers_oklab)
    print(f"  Found {n_clusters} cluster(s) in downscaled image.")

    original_arr = np.array(img_for_quant, dtype=np.uint8)
    original_float = original_arr.astype(np.float64) / 255.0
    original_xyz = colour.sRGB_to_XYZ(original_float)
    original_oklab = colour.XYZ_to_Oklab(original_xyz)
    orig_pixels = original_oklab.reshape(-1, 3)

    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centers_oklab)
    distances, indices = nn.kneighbors(orig_pixels)
    assigned_oklab = centers_oklab[indices.flatten()].reshape(orig_h, orig_w, 3)

    assigned_xyz = colour.Oklab_to_XYZ(assigned_oklab)
    assigned_srgb = colour.XYZ_to_sRGB(assigned_xyz)
    assigned_srgb = np.clip(assigned_srgb, 0, 1)
    quantized_full = (assigned_srgb * 255).astype(np.uint8)

    quantized_full_img = Image.fromarray(quantized_full, mode="RGB")

    if has_alpha and mask is not None:
        quantized_full_img = quantized_full_img.convert("RGBA")
        quantized_full_img.putalpha(Image.fromarray(mask, mode="L"))

    return quantized_full_img


def get_palette_oklab(palette_img: Image.Image) -> np.ndarray:
    """
    Convert the given palette image to Oklab color space and return an
    N x 3 numpy array of the unique colors in that palette.
    """
    pal_rgb = palette_img.convert("RGB")
    pal_pixels = np.array(pal_rgb).reshape(-1, 3)
    unique_colors = np.unique(pal_pixels, axis=0)
    unique_float = unique_colors.astype(np.float64) / 255.0
    xyz = colour.sRGB_to_XYZ(unique_float)
    oklab = colour.XYZ_to_Oklab(xyz)
    return oklab


def determine_som_grid(k: int) -> tuple:
    """
    Determine an approximate square grid (rows, cols) for a SOM that
    has exactly k nodes total.
    """
    rows = int(math.floor(math.sqrt(k)))
    cols = int(math.ceil(k / rows))
    return rows, cols


def som_quantize_with_palette(image: Image.Image, palette_img: Image.Image, iterations: int = 71) -> Image.Image:
    """
    Quantize 'image' so that its final colors are adapted from the given palette.
    
    Steps:
      1. Convert both the image and palette to Oklab.
      2. Determine a SOM grid whose number of nodes is equal to the number of palette colors.
      3. Initialize the SOM nodes with the palette colors.
      4. Train the SOM on the image's pixel data (in Oklab) for a number of iterations.
      5. For each image pixel, find its best matching unit (BMU) in the SOM.
      6. Optionally, snap each BMU’s weight vector to the nearest original palette color.
      7. Convert the quantized image back to sRGB.

    This produces an image whose colors come from a slightly adapted version
    of your target palette.
    """
    # --- Convert image to Oklab.
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb, dtype=np.uint8)
    float_arr = arr.astype(np.float64) / 255.0
    xyz_arr = colour.sRGB_to_XYZ(float_arr)
    oklab_arr = colour.XYZ_to_Oklab(xyz_arr)
    pixels = oklab_arr.reshape(-1, 3)

    # --- Quick check to make sure we're not trying to quantize a bajillion colors
    num_colors = len(palette_img.getcolors(16777216))
    if num_colors > 256:
        palette_img = palette_img.quantize(colors=256, method=2, kmeans=256, dither=0).convert("RGB")
    
    # --- Convert palette to Oklab.
    palette_oklab = get_palette_oklab(palette_img)  # shape (k, 3)
    k = len(palette_oklab)
    if k == 0:
        print("Warning: Palette image contains no colors!")
        return img_rgb
    print(f"Using a palette of {k} unique color(s).")
    
    # --- Determine SOM grid shape: use exactly k nodes.
    rows, cols = determine_som_grid(k)
    total_nodes = rows * cols
    print(f"Initializing SOM grid of size {rows}x{cols} (total {total_nodes} nodes).")
    
    # --- Initialize SOM with the palette colors.
    init_weights = np.zeros((rows, cols, 3))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            init_weights[i, j] = palette_oklab[idx % k]
    
    # --- Create and initialize the SOM.
    som = MiniSom(rows, cols, 3, sigma=0.22, learning_rate=0.2, random_seed=42)
    som._weights = init_weights.copy()
    
    # --- Train the SOM on the image's Oklab pixels.
    print("Training SOM...")
    som.train_random(pixels, iterations)
    
    # --- Snap each SOM node to the nearest original palette color.
    weights = som._weights.reshape(-1, 3)  # shape (total_nodes, 3)
    nn_pal = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(palette_oklab)
    _, indices = nn_pal.kneighbors(weights)
    snapped_weights = palette_oklab[indices.flatten()].reshape(rows, cols, 3)
    
    # --- Quantize full-resolution image: assign each pixel to its BMU.
    H, W, _ = oklab_arr.shape
    quantized_oklab = np.zeros_like(oklab_arr)
    for i in range(H):
        for j in range(W):
            pixel = oklab_arr[i, j]
            bmu = som.winner(pixel)
            quantized_oklab[i, j] = snapped_weights[bmu]
    
    # --- Convert quantized Oklab image back to sRGB.
    quant_xyz = colour.Oklab_to_XYZ(quantized_oklab)
    quant_srgb = colour.XYZ_to_sRGB(quant_xyz)
    quant_srgb = np.clip(quant_srgb, 0, 1)
    out_arr = (quant_srgb * 255).astype(np.uint8)
    return Image.fromarray(out_arr, mode="RGB")


def classify_rgb_pixel(pixel):
    """
    0 => black(0,0,0)
    1 => white(255,255,255)
    2 => magenta(255,0,255)
    3 => everything else
    """
    if pixel == (0, 0, 0):
        return 0
    elif pixel == (255, 255, 255):
        return 1
    elif pixel == (255, 0, 255):
        return 2
    else:
        return 3


def blend_images_with_mask(image1, image2, mask):
    """
    Blends two RGBA images using a single-channel (L) mask.
    """
    if image1.size != mask.size or image2.size != mask.size:
        raise ValueError("All images (image1, image2, mask) must have the same dimensions.")

    arr1 = np.array(image1.convert("RGBA"), dtype=np.float32)
    arr2 = np.array(image2.convert("RGBA"), dtype=np.float32)
    mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0
    mask_arr = np.expand_dims(mask_arr, axis=-1)

    blended = arr1 * mask_arr + arr2 * (1 - mask_arr)
    blended = blended.clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGBA")


def tile_has_magenta(tile_rgb: Image.Image) -> bool:
    """Returns True if the tile has at least one (255, 0, 255) pixel."""
    return (255, 0, 255) in tile_rgb.getdata()


def add_noise_to_feather(mask: Image.Image, noise_level=10) -> Image.Image:
    """
    Adds random small perturbations to the middle range of a feathered mask,
    to avoid harsh edges.
    """
    arr = np.array(mask, dtype=np.int16)
    non_extremes = (arr > 0) & (arr < 255)
    noise = np.random.randint(-noise_level, noise_level + 1, arr.shape) * 2
    arr[non_extremes] += noise[non_extremes]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def determine_tile_size_from_master(master_mask_rgb: Image.Image):
    """
    Tries to guess the tile size by scanning how far black(0) extends
    from the top-left corner horizontally and vertically.
    """
    width, height = master_mask_rgb.size

    tile_width = 0
    for x in range(width):
        if classify_rgb_pixel(master_mask_rgb.getpixel((x, 0))) == 0:
            tile_width += 1
        else:
            break

    tile_height = 0
    for y in range(height):
        if classify_rgb_pixel(master_mask_rgb.getpixel((0, y))) == 0:
            tile_height += 1
        else:
            break

    return (tile_width, tile_height)

