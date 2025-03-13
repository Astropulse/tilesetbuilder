import requests
import base64
import io
import math
import random
import time
import os
import numpy as np
from PIL import Image, ImageFilter
import gradio as gr

from concurrent.futures import ThreadPoolExecutor, as_completed

from util.utils import (
    som_quantize_with_palette,
    classify_rgb_pixel,
    blend_images_with_mask,
    tile_has_magenta,
    add_noise_to_feather,
    determine_tile_size_from_master
)

# =============================================================================
# TILE GENERATION AND SEAM MASKS
# =============================================================================

def create_seam_mask_from_tile_color(tile_rgb, seam_width, feather_radius):
    """
    Builds a mask marking boundaries between pixel labels in a tile.
    Then expands and feathers that boundary if requested.
    """
    w, h = tile_rgb.size
    labels_2d = []
    px_rgb = tile_rgb.load()

    # A: label each pixel
    for y in range(h):
        row = []
        for x in range(w):
            row.append(classify_rgb_pixel(px_rgb[x,y]))
        labels_2d.append(row)
    label_arr = np.array(labels_2d, dtype=np.uint8)

    # B: detect boundary pixels
    seam_arr = np.zeros((h, w), dtype=np.uint8)
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    for y in range(h):
        for x in range(w):
            my_label = label_arr[y, x]
            for dy, dx in directions:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and label_arr[ny, nx] != my_label:
                    seam_arr[y, x] = 255
                    break

    seam_img = Image.fromarray(seam_arr, mode="L")

    # C: morphological expansion (MaxFilter) if seam_width > 1
    seam_width = int(round(seam_width))
    if seam_width < 1:
        seam_width = 1
    if seam_width % 2 == 0:
        seam_width += 1
    if seam_width > 1:
        seam_img = seam_img.filter(ImageFilter.MaxFilter(seam_width))

    # D: feather with GaussianBlur
    if feather_radius > 0:
        seam_img = seam_img.filter(ImageFilter.GaussianBlur(feather_radius))

    return seam_img


def stitch_tiles(tiles, grid_size, tile_size, debug_mode=False):
    rows, cols = grid_size
    stitched = Image.new("RGBA", (cols * tile_size[0], rows * tile_size[1]))

    if isinstance(tiles, dict):
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                tile = tiles[(r, c)]
                box = ((c - 1) * tile_size[0], (r - 1) * tile_size[1])
                stitched.paste(tile, box)
    else:
        idx = 0
        for r in range(rows):
            for c in range(cols):
                tile = tiles[idx]
                box = (c * tile_size[0], r * tile_size[1])
                stitched.paste(tile, box)
                idx += 1
    
    if debug_mode:
        stitched.save("debug/debug_stitched_tiles.png")
    return stitched


def apply_seam_replacements(
    tileset_image,
    seams_image,
    master_mask,
    grid_size,
    tile_size,
    seam_width,
    feather_radius,
    debug_mode=False
):
    """
    1) For each tile, build a "raw" seam mask from the tile's region in master_mask.
    2) Stitch all tile masks into one large 'combined_seam_masks' and globally normalize.
    3) Re-split into tiles, use each sub-mask to blend seam_image over tileset_image.
    """
    rows, cols = grid_size
    final_image = tileset_image.copy()
    seam_masks = {}

    # (A) Build raw seam mask for each tile
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            left = (c - 1) * tile_size[0]
            upper = (r - 1) * tile_size[1]
            box = (left, upper, left + tile_size[0], upper + tile_size[1])

            tile_mask_rgb = master_mask.crop(box).convert("RGB")
            extrema = tile_mask_rgb.getextrema()
            all_mins = [pair[0] for pair in extrema]
            all_maxes = [pair[1] for pair in extrema]
            # If uniform => no seam
            if min(all_mins) == max(all_maxes):
                seam_mask = Image.new("L", tile_size, 0)
            else:
                seam_mask = create_seam_mask_from_tile_color(
                    tile_mask_rgb, seam_width, feather_radius
                )
            seam_masks[(r, c)] = seam_mask

    # (B) Stitch raw seam masks into one large mask
    combined_masks = stitch_tiles(seam_masks, grid_size, tile_size)
    combined_masks = combined_masks.convert("L")

    # (C) Globally normalize
    arr = np.array(combined_masks, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = ((arr - mn) / (mx - mn)) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    normalized_combined = Image.fromarray(arr, mode="L")
    if debug_mode:
        normalized_combined.save("debug/debug_global_normalized_seam_mask.png")

    # (D) Re-split normalized mask and blend
    normed_masks = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            left = (c - 1) * tile_size[0]
            upper = (r - 1) * tile_size[1]
            box = (left, upper, left + tile_size[0], upper + tile_size[1])
            normed_masks[(r, c)] = normalized_combined.crop(box)

    # (E) Blend each tile
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            left = (c - 1) * tile_size[0]
            upper = (r - 1) * tile_size[1]
            box = (left, upper, left + tile_size[0], upper + tile_size[1])

            tile_region = tileset_image.crop(box)
            seam_region = seams_image.crop(box)
            seam_mask = normed_masks[(r, c)]

            blended_tile = blend_images_with_mask(seam_region, tile_region, seam_mask)
            final_image.paste(blended_tile, box)

    if debug_mode:
        final_image.save("debug/debug_final_seamed_tileset.png")

    return final_image


# =============================================================================
# PARTIAL TILE BLENDING
# =============================================================================

def partial_tile_black_white_then_magenta(mask_rgb,
                                          outside_img,
                                          inside_img,
                                          feather_radius=5,
                                          noise_level=10):
    """
    Same partial-tile code you already have, plus a simple neighbor-cleanup 
    pass that:
      - looks only at the originally magenta region (label=2),
      - finds pixels that ended up non-magenta, 
      - counts how many of their 8 neighbors in that region have the same color,
      - if fewer than 3 neighbors match => convert pixel to magenta.
    """
    w, h = mask_rgb.size
    
    # -----------------------------------------------------------
    # Step A: black vs white (ignore magenta => black)
    # -----------------------------------------------------------
    submask_bw = Image.new("L", (w, h), 0)
    px_mask = mask_rgb.load()
    px_bw = submask_bw.load()

    for y in range(h):
        for x in range(w):
            label = classify_rgb_pixel(px_mask[x,y])
            px_bw[x,y] = 255 if label == 1 else 0

    submask_bw = submask_bw.filter(ImageFilter.GaussianBlur(feather_radius))
    if noise_level > 0:
        submask_bw = add_noise_to_feather(submask_bw, noise_level)

    bw_arr = np.array(submask_bw, dtype=np.float32) / 255.0

    outside_resized = outside_img.resize((w, h), Image.Resampling.NEAREST).convert("RGB")
    inside_resized  = inside_img.resize((w, h),  Image.Resampling.NEAREST).convert("RGB")

    out_arr = np.array(outside_resized, dtype=np.float32)
    in_arr  = np.array(inside_resized,  dtype=np.float32)

    stepA_arr = out_arr*(1 - bw_arr[...,None]) + in_arr*(bw_arr[...,None])
    stepA_arr = np.clip(stepA_arr, 0, 255).astype(np.uint8)

    # -----------------------------------------------------------
    # Step B: Randomly dither for magenta(2)
    # -----------------------------------------------------------
    # 1) 0/255 mask for magenta
    mask_m = Image.new("L", (w, h), 0)
    px_m = mask_m.load()
    for y in range(h):
        for x in range(w):
            if classify_rgb_pixel(px_mask[x,y]) == 2:
                px_m[x,y] = 255

    # 2) Feather + noise => fractional probability
    mask_m = mask_m.filter(ImageFilter.GaussianBlur(feather_radius/3))
    if noise_level > 0:
        mask_m = add_noise_to_feather(mask_m, noise_level)

    m_arr = np.array(mask_m, dtype=np.float32) / 255.0

    # 3) For each pixel => magenta if random < m_arr
    random_map = np.random.random((h, w))
    stepA_float = stepA_arr.astype(np.float32)
    magenta_arr = np.full((h, w, 3), [255,0,255], dtype=np.float32)

    magenta_mask = (random_map < m_arr)
    final_arr = np.empty((h, w, 3), dtype=np.float32)
    final_arr[ magenta_mask ] = magenta_arr[ magenta_mask ]
    final_arr[~magenta_mask ] = stepA_float[~magenta_mask]
    final_arr = np.clip(final_arr, 0, 255).astype(np.uint8)

    # -----------------------------------------------------------
    # Cleanup: ONLY in originally magenta region (label=2).
    # If a pixel ended up non-magenta, but doesn't have at least
    # 3 neighbors of the same color, set it to magenta.
    # -----------------------------------------------------------
    # Build a boolean array: True => originally label=2
    orig_mag = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            if classify_rgb_pixel(px_mask[x, y]) == 2:
                orig_mag[y, x] = True

    # Convert final_arr to a PIL image for easy neighbor checking
    final_img = Image.fromarray(final_arr, mode="RGB")
    px_final = final_img.load()  # so px_final[x, y] => (R,G,B)

    neighbors_8 = [(-1,-1), (-1,0), (-1,1),
                   (0,-1),          (0,1),
                   (1,-1), (1,0),  (1,1)]

    MAGENTA = (255,0,255)
    def same_color(a, b):
        return (a[0]==b[0]) and (a[1]==b[1]) and (a[2]==b[2])

    # We'll make a copy so changes don't cause chain reactions
    filtered_arr = final_arr.copy()

    for y in range(h):
        for x in range(w):
            if not orig_mag[y, x]:
                # Not originally magenta => skip
                continue

            current_color = px_final[x, y]
            if same_color(current_color, MAGENTA):
                # Already magenta => skip
                continue

            # Count how many neighbors in the *same region* 
            # share the same color
            same_count = 0
            for dy, dx in neighbors_8:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    # We only consider neighbors that are also originally magenta
                    if orig_mag[ny, nx]:
                        neighbor_color = px_final[nx, ny]
                        if same_color(neighbor_color, current_color):
                            same_count += 1

            # If fewer than 3 neighbors match => convert to magenta
            if same_count < 3:
                filtered_arr[y, x] = [255,0,255]

    # Return final cleaned image
    return Image.fromarray(filtered_arr.astype(np.uint8), mode="RGB")


def generate_tileset_from_master_mask(
    master_mask_rgb,
    outside_img,
    inside_img,
    feather_radius=5,
    noise_level=10,
    debug_mode=False
):
    """
    Generates the base set of tiles from a master mask image. Each tile
    is either uniform or partially blended using black/white/magenta logic.
    """
    tile_size = determine_tile_size_from_master(master_mask_rgb)
    tw, th = tile_size
    grid_cols = master_mask_rgb.width // tw
    grid_rows = master_mask_rgb.height // th

    tiles = {}
    for row in range(1, grid_rows+1):
        for col in range(1, grid_cols+1):
            left = (col - 1)*tw
            upper = (row - 1)*th
            box = (left, upper, left+tw, upper+th)
            tile_mask = master_mask_rgb.crop(box)

            # Classify entire tile
            labels = [classify_rgb_pixel(tile_mask.getpixel((x, y))) 
                      for y in range(th) for x in range(tw)]
            unique_labels = set(labels)

            if len(unique_labels) == 1:
                # Uniform tile
                only = unique_labels.pop()
                if only == 0:
                    tile_img = outside_img.resize(tile_size, Image.Resampling.NEAREST).convert("RGB")
                elif only == 1:
                    tile_img = inside_img.resize(tile_size, Image.Resampling.NEAREST).convert("RGB")
                elif only == 2:
                    tile_img = Image.new("RGB", tile_size, (255,0,255))
                else:  # label=3 => treat as black
                    tile_img = outside_img.resize(tile_size, Image.Resampling.NEAREST).convert("RGB")
            else:
                # Partial tile => black/white, then magenta
                tile_img = partial_tile_black_white_then_magenta(
                    tile_mask, outside_img, inside_img, feather_radius, noise_level
                )

            if debug_mode:
                tile_img.save(f"debug/debug_tile_{row}_{col}.png")
            tiles[(row,col)] = tile_img

    return tiles, (grid_rows, grid_cols), tile_size


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def generate_images(
    api_key,
    prompt,
    input_image=None,
    strength=0.5,
    style="default",
    model="RD_FLUX",
    width=256,
    height=256,
    num_images=1,
    seed=0
):
    """
    Example stub function calling an external API (e.g. RetroDiffusion).
    Modify as needed for your actual endpoint.
    """
    if input_image is not None:
        buf = io.BytesIO()
        input_image.convert("RGB").save(buf, format="PNG")
        base64_input_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    url = "https://api.retrodiffusion.ai/v1/inferences"
    headers = {"X-RD-Token": api_key}

    if input_image is not None:
        payload = {
            "prompt": prompt,
            "prompt_style": style,
            "model": model,
            "width": width,
            "height": height,
            "input_image": base64_input_image,
            "strength": strength,
            "num_images": num_images,
            "seed": seed,
        }
    else:
        payload = {
            "prompt": prompt,
            "prompt_style": style,
            "model": model,
            "width": width,
            "height": height,
            "num_images": num_images,
            "seed": seed,
        }

    response = requests.post(url, headers=headers, json=payload)
    images = []
    if response.status_code == 200:
        data = response.json()
        base64_images = data.get("base64_images", [])
        if base64_images:
            for img_data in base64_images:
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
        else:
            print("No images returned by the API.")
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")

    return images


def process_tile(
    row,
    col,
    tiles,
    tile_size,
    master_mask,
    outside_img,
    api_key,
    outside_prompt,
    inside_prompt,
    outside_texture,
    debug_mode=False
):
    """
    Checks if a tile is uniform; if not, calls the generation API to build
    a "seam replacement" image. Returns (row, col, final_image_for_that_tile).
    """
    tile = tiles[(row, col)]
    left = (col - 1) * tile_size[0]
    upper = (row - 1) * tile_size[1]
    box = (left, upper, left + tile_size[0], upper + tile_size[1])

    # Check uniform
    tile_rgb = master_mask.crop(box).convert("RGB")
    extrema = tile_rgb.getextrema()
    min_vals = [c[0] for c in extrema]
    max_vals = [c[1] for c in extrema]
    if min(min_vals) == max(max_vals):
        # Uniform => skip generation
        if debug_mode:
            tile.save(f"debug/debug_generated_tile_{row}_{col}.png")
        print(f"Skipping seam generation for tile {row}:{col} (uniform).")
        return (row, col, tile)
    
    # Build prompt (outside on top of inside, plus magenta if needed)
    prompt = f"{outside_prompt} on top of {inside_prompt}"
    if tile_has_magenta(tile_rgb):
        prompt += " on a magenta background"

    # Generate images
    images = generate_images(
        api_key=api_key,
        prompt=prompt,
        input_image=tile,
        strength=0.4,
        model="RD_FLUX",
        style="mc_texture",
        width=outside_texture.width,
        height=outside_texture.height,
        num_images=1,
        seed=random.randint(1, 999999)
    )
    print(f"Generated seam replacement for tile {row}:{col} [prompt='{prompt}']")

    # Post-process: downsize & SOM-quantize w/ tile as palette
    for image in images:
        factor = 8
        new_width = image.width // factor
        new_height = image.height // factor
        image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
        image = som_quantize_with_palette(image, tile)

        if debug_mode:
            image.save(f"debug/debug_generated_tile_{row}_{col}.png")

        return (row, col, image)


def run_pipeline(
    api_key,
    outside_prompt,
    inside_prompt,
    outside_texture,
    inside_texture,
    master_mask_choice,
    debug_mode
):
    """
    High-level pipeline to:
    1) Load the mask and generate base tiles
    2) Generate partial seam tiles
    3) Blend seams
    4) Combine palette from outside+inside (and magenta if needed)
    5) Final SOM quantize
    6) Output final
    """
    start_time = time.time()
    if outside_texture.size != inside_texture.size:
        raise ValueError("Outside and inside textures must be the same size.")

    w, h = outside_texture.size
    if w < 16 or h < 16 or w > 128 or h > 128:
        raise ValueError("Textures must be between 16x16 and 128x128.")

    mask_path = os.path.join("mask", master_mask_choice)
    master_mask_rgb = Image.open(mask_path).convert("RGB")

    # 1) Determine the original tile size from the master mask
    orig_tile_size = determine_tile_size_from_master(master_mask_rgb)
    grid_cols = master_mask_rgb.width  // orig_tile_size[0]
    grid_rows = master_mask_rgb.height // orig_tile_size[1]

    # 2) If the master mask’s tile size is different from the texture size (w,h), 
    #    we rescale the master mask to match. Each tile becomes exactly w x h.
    if orig_tile_size != (w, h):
        new_width  = grid_cols * w
        new_height = grid_rows * h
        print(f"Resizing master mask from {master_mask_rgb.size} to {(new_width, new_height)}...")
        master_mask_rgb = master_mask_rgb.resize((new_width, new_height), Image.Resampling.NEAREST)

    # Now each tile is guaranteed to be (w, h).
    tile_size = (w, h)

    feather_radius = math.sqrt(w*h) / 14
    tiles, grid_size, _ = generate_tileset_from_master_mask(
        master_mask_rgb, outside_texture, inside_texture, 
        feather_radius, debug_mode=debug_mode
    )
    tiles, grid_size, _ = generate_tileset_from_master_mask(
        master_mask_rgb, outside_texture, inside_texture, feather_radius, debug_mode=debug_mode
    )
    raw_tileset = stitch_tiles(tiles, grid_size, tile_size, debug_mode=debug_mode)

    # Generate seam replacements (partial tiles)
    generated_tiles = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for r in range(1, grid_size[0]+1):
            for c in range(1, grid_size[1]+1):
                f = executor.submit(
                    process_tile,
                    r, c, tiles, tile_size, master_mask_rgb,
                    outside_texture, api_key,
                    outside_prompt, inside_prompt,
                    outside_texture, debug_mode
                )
                futures.append(f)

        for future in as_completed(futures):
            row, col, tile_image = future.result()
            generated_tiles[(row, col)] = tile_image
    
    generated_tileset = stitch_tiles(generated_tiles, grid_size, tile_size, debug_mode=debug_mode)

    # Apply seam replacements
    seam_feather = math.sqrt(w*h) / 10
    final_tileset = apply_seam_replacements(
        raw_tileset,
        generated_tileset,
        master_mask_rgb.convert("L"),
        grid_size,
        tile_size,
        int(seam_feather),
        seam_feather,
        debug_mode=debug_mode
    ).convert("RGB")

    # Build combined palette from outside+inside (and magenta if mask has any)
    combined = Image.new("RGB", (w + w, max(h, h)), (0, 0, 0))
    combined.paste(outside_texture.convert("RGB"), (0, 0))
    combined.paste(inside_texture.convert("RGB"), (w, 0))
    if (255, 0, 255) in master_mask_rgb.getdata():
        combined.putpixel((0, 0), (255,0,255))

    # Final quantize
    final_tileset = som_quantize_with_palette(final_tileset, combined)

    # 2) Convert any pure magenta pixel to alpha=0
    rgba_img = final_tileset.convert("RGBA")
    arr = np.array(rgba_img, dtype=np.uint8)  # shape (H, W, 4)
    magenta_mask = (arr[...,0] == 255) & (arr[...,1] == 0) & (arr[...,2] == 255)
    arr[magenta_mask, 3] = 0

    # Convert back to a PIL Image
    final_tileset = Image.fromarray(arr, mode="RGBA")

    # Save final outputs
    final_tileset.save("final_tileset.png")
    upscale_factor = 8
    final_upscaled = final_tileset.resize(
        (final_tileset.width * upscale_factor, final_tileset.height * upscale_factor),
        Image.Resampling.NEAREST
    )

    elapsed = time.time() - start_time
    print(f"Total process time: {elapsed:.2f} seconds")
    return final_upscaled, "final_tileset.png"


# =============================================================================
# GRADIO UI HELPERS
# =============================================================================

def preview_mask_and_credit(mask_filename):
    """
    Quick function: loads the mask, displays it,
    and calculates how many partial tiles would cost a "credit."
    """
    mask_path = os.path.join("mask", mask_filename)
    mask_img = Image.open(mask_path).convert("RGB")

    tile_size = determine_tile_size_from_master(mask_img)
    grid_cols = mask_img.width // tile_size[0]
    grid_rows = mask_img.height // tile_size[1]

    credit_count = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            left = col * tile_size[0]
            upper = row * tile_size[1]
            tile_rgb = mask_img.crop((left, upper, left+tile_size[0], upper+tile_size[1]))

            # Identify unique labels
            labels = [classify_rgb_pixel(tile_rgb.getpixel((x, y))) 
                      for y in range(tile_rgb.height) for x in range(tile_rgb.width)]
            unique_l = set(labels)

            # If it's 1 label in {0,1,2}, no credit
            if len(unique_l) == 1:
                only = list(unique_l)[0]
                if only not in (0,1,2):
                    credit_count += 1
            else:
                credit_count += 1

    return mask_img, f"Credits required: {credit_count}"


def refresh_mask_folder():
    """
    Fetches new images in the 'mask' folder, updates dropdown.
    """
    new_masks = [f for f in os.listdir("mask") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if new_masks:
        return gr.update(choices=new_masks, value=new_masks[0])
    else:
        return gr.update(choices=[], value=None)


# =============================================================================
# GRADIO UI
# =============================================================================

mask_files = [f for f in os.listdir("mask") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if mask_files:
    mask_preview_default, credit_info_default = preview_mask_and_credit(mask_files[0])
else:
    mask_preview_default, credit_info_default = None, "No mask files available"

with gr.Blocks(title="Tileset Generator") as demo:
    gr.Markdown("""
    # Tileset Generator

    Create tilesets from two tiles and a tileset mask
    """)
    with gr.Row():
        with gr.Column():
            api_key_in = gr.Textbox(
                label="Retro Diffusion API Key",
                value=open("util/api_key.txt").read().strip() if os.path.exists("util/api_key.txt") else "",
                placeholder="Enter your API key"
            )
            gr.Markdown("""
            Get your key from [the developer tools section](https://www.retrodiffusion.ai/app/devtools)
            """)
            outside_prompt_in = gr.Textbox(label="Outside Prompt", value="")
            inside_prompt_in = gr.Textbox(label="Inside Prompt", value="")
            outside_texture_in = gr.Image(label="Outside Texture (black area)", type="pil")
            inside_texture_in = gr.Image(label="Inside Texture (white area)", type="pil")

            gr.Markdown("""
            For more information on Master Masks and how to make your own go [here](https://github.com/Astropulse/tilesetbuilder?tab=readme-ov-file#adding-master-masks)
            """)

            master_mask_dropdown = gr.Dropdown(
                choices=mask_files,
                label="Master Mask (RGB with black/white/magenta)",
                value=mask_files[0] if mask_files else None
            )
            refresh_masks_btn = gr.Button("Refresh Mask Folder")
            debug_mode_in = gr.Checkbox(label="Debug Mode", value=False)
            run_btn = gr.Button("Generate Tileset")

        with gr.Column():
            mask_preview = gr.Image(label="Master Mask Preview", value=mask_preview_default)
            credit_info = gr.Textbox(label="Credit Cost", value=credit_info_default)
            final_tileset_out = gr.Image(label="Final Tileset (Upscaled)")
            download_file_out = gr.File(label="Save Final Tileset")

    master_mask_dropdown.change(
        fn=preview_mask_and_credit,
        inputs=master_mask_dropdown,
        outputs=[mask_preview, credit_info]
    )
    refresh_masks_btn.click(fn=refresh_mask_folder, inputs=[], outputs=master_mask_dropdown)
    
    run_btn.click(
        fn=run_pipeline,
        inputs=[
            api_key_in,
            outside_prompt_in,
            inside_prompt_in,
            outside_texture_in,
            inside_texture_in,
            master_mask_dropdown,
            debug_mode_in
        ],
        outputs=[final_tileset_out, download_file_out]
    )

demo.launch(inbrowser=True)
