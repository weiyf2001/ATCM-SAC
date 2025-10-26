import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def compute_distance_field(grid, start):
    """
    Compute the BFS-based distance field from a given start point on a binary grid.

    Parameters:
    - grid: 2D numpy array where 0 = sea (traversable), 1 = land (obstacle).
    - start: Tuple (row, col) indicating the starting point (must be on sea, i.e., grid[start] == 0).

    Returns:
    - Normalized distance field (distances divided by 300.0) with np.inf where unreachable or on land.
    """
    rows, cols = grid.shape
    distance = np.full((rows, cols), np.inf)
    visited = np.zeros_like(grid, dtype=bool)

    queue = deque([start])
    distance[start] = 0
    visited[start] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] == 0 and not visited[nr, nc]:
                    distance[nr, nc] = distance[r, c] + 1
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    return distance / 300.0


def crop_centered_subregion_with_offset(array, center, crop_size):
    """
    Crop a subregion of given size centered at `center`. Returns the cropped array
    and the offset (top-left corner) of the crop within the original array.

    Handles boundary clipping gracefully.
    """
    h, w = crop_size
    r, c = center

    dh, dw = h // 2, w // 2

    r_min = max(0, r - dh)
    r_max = min(array.shape[0], r + h - dh)
    c_min = max(0, c - dw)
    c_max = min(array.shape[1], c + w - dw)

    cropped = array[r_min:r_max, c_min:c_max]

    # Compute where the center lands in the cropped array
    center_in_crop = (r - r_min, c - c_min)

    return cropped, center_in_crop


def generate_bfs_distance_visualization(
        mask_path,
        sub_region_size=(120, 210),
        start_x=150,
        start_y=230,
        selected_local_point=(80, 40),
        local_target_point=(60, 100),
        sub_sub_size=(12, 21)
):
    """
    Full pipeline: load mask → extract sub-region → BFS → visualize → crop sub-sub → tensor conversion.

    Returns:
    - D_G: torch.Tensor of shape (sub_h, sub_w)
    - D_L: torch.Tensor of actual cropped shape (may be smaller near edges)
    """
    # Load mask
    print(f"Loading land-sea mask from: {mask_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    land_sea_mask = np.load(mask_path)
    print(f"Global mask shape: {land_sea_mask.shape}")

    sub_h, sub_w = sub_region_size
    height, width = land_sea_mask.shape

    if start_x + sub_h > height or start_y + sub_w > width:
        raise ValueError("Sub-region exceeds global mask dimensions.")

    sub_mask = land_sea_mask[start_x:start_x + sub_h, start_y:start_y + sub_w]

    if sub_mask[selected_local_point] != 0:
        raise ValueError("Start point is on land.")

    # Compute distance field
    distance_field = compute_distance_field(sub_mask, selected_local_point)

    # --- Visualization ---
    # Fig 1: Global mask with sub-region box
    plt.figure(figsize=(12, 9))
    masked_global = np.where(land_sea_mask == 1, np.nan, land_sea_mask)
    plt.imshow(masked_global, cmap='viridis', origin='lower', interpolation='none')
    rect_x = [start_y, start_y + sub_w, start_y + sub_w, start_y, start_y]
    rect_y = [start_x, start_x, start_x + sub_h, start_x + sub_h, start_x]
    plt.plot(rect_x, rect_y, 'r--', linewidth=2)
    plt.title('Global Land-Sea Mask with Selected Sub-region')
    plt.axis('off')
    plt.show()

    # Fig 2: Sub-region with both points
    plt.figure(figsize=(12, 9))
    masked_sub = np.where(sub_mask == 1, np.nan, sub_mask)
    plt.imshow(masked_sub, cmap='viridis', origin='lower', interpolation='none')
    plt.scatter(selected_local_point[1], selected_local_point[0], c='red', marker='o', s=100, label='Target')
    plt.scatter(local_target_point[1], local_target_point[0], c='white', marker='o', s=100, linewidths=2, label='Start')
    plt.title('Sub-region Mask with Start and Target Points')
    plt.legend()
    plt.axis('off')
    plt.show()

    # Fig 3: Full BFS distance field with target
    plt.figure(figsize=(10, 8))
    im = plt.imshow(distance_field, cmap='viridis', origin='lower', interpolation='none')
    plt.scatter(selected_local_point[1], selected_local_point[0], c='red', marker='o', s=100, label='Target')
    plt.scatter(local_target_point[1], local_target_point[0], c='white', marker='o', s=100, linewidths=3,
                label='Start')
    plt.title('BFS Distance Field (Sub-region)')
    plt.axis('off')
    plt.legend()
    cbar = plt.colorbar(im, shrink=0.6)
    cbar.set_label('Normalized Distance to Start Point')
    plt.show()

    # Crop sub-sub region and get center location within it
    sub_sub_region, center_in_subsub = crop_centered_subregion_with_offset(
        distance_field, local_target_point, sub_sub_size
    )
    print(f"Sub-sub region shape: {sub_sub_region.shape} (requested: {sub_sub_size})")

    # Fig 4: Sub-sub region (local crop)
    plt.figure(figsize=(6, 4))
    im4 = plt.imshow(sub_sub_region, cmap='viridis', origin='lower', interpolation='none')
    plt.scatter(center_in_subsub[1], center_in_subsub[0], c='white', marker='o', s=100, linewidths=3,
                label='Start')
    plt.title('Sub-sub Region: Local Distance Crop')
    plt.axis('off')
    plt.legend()
    cbar4 = plt.colorbar(im4, shrink=0.7)
    cbar4.set_label('Normalized Distance')
    plt.tight_layout()
    plt.show()

    # Convert to tensors
    D_G = torch.from_numpy(distance_field).float()
    D_L = torch.from_numpy(sub_sub_region).float()

    print(f"D_G tensor shape: {D_G.shape}")
    print(f"D_L tensor shape: {D_L.shape}")
    return D_G, D_L


# -------------------------------
if __name__ == "__main__":
    mask_path = './land_sea_mask.npy'
    D_G, D_L = generate_bfs_distance_visualization(
        mask_path=mask_path,
        sub_region_size=(120, 210),
        start_x=150,
        start_y=230,
        selected_local_point=(80, 40),
        local_target_point=(60, 100),
        sub_sub_size=(12, 21)
    )