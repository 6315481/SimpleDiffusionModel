import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_cdf(X):
    # 入力データをソートする
    X_sorted = np.sort(X, axis=0)

    # y軸の値を計算する（0から1までの間で等間隔に）
    y = np.arange(1, len(X_sorted) + 1) / len(X_sorted)

    # CDFをプロットする
    plt.plot(X_sorted, y, marker='.', linestyle='none')
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function (CDF)')
    plt.grid(True)
    plt.show()

def plot_pdf(X, bw='scott', kernel='gau', **kwargs):
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(X, bw=bw, kernel=kernel, **kwargs)
    ax.set_title("Probability Density Function (PDF) with KDE")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_paths(X):
    """
    X: ndarray of shape [n, T]
       n is the number of samples, T is the number of time steps
    """
    n, T = X.shape
    
    # 時間軸の値
    t = np.arange(T)
    
    # 各サンプルのパスをプロット
    for i in range(n):
        plt.plot(t, X[i, :])
    
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Time Paths for Each Sample')
    plt.show()

def plot_2d_path(a, interpolate=True):
    plt.figure(figsize=(8, 6))
    
    if interpolate:
        plt.plot(a[:, 0], a[:, 1], marker='o')
    else:
        plt.scatter(a[:, 0], a[:, 1], marker='o')
        
    plt.grid(True)
    plt.show()

def plot_2d_paths(a, interpolate=True):
    N = a.shape[0]
    rows = (N + 4) // 5

    fig, axs = plt.subplots(rows, 5, figsize=(15, 3*rows))

    for i in range(N):
        ax = axs[i // 5, i % 5]
        
        if interpolate:
            ax.plot(a[i, :, 0], a[i, :, 1], marker='o')
        else:
            ax.scatter(a[i, :, 0], a[i, :, 1], marker='o')
        
        ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_2d_paths_combined(a, interpolate=True):
    N = a.shape[0]
    
    colors = plt.cm.jet(np.linspace(0, 1, N))
    
    plt.figure(figsize=(10, 8))
    for i in range(N):
        path = a[i]
        
        if interpolate:
            plt.plot(path[:, 0], path[:, 1], color=colors[i], linestyle='-', linewidth=2)
        else:
            plt.scatter(path[:, 0], path[:, 1], color=colors[i], marker='o')
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Combined 2D Paths Plot')
        plt.grid(True)
    plt.legend()
    plt.show()

def display_images(images):
    """
    Display images in a grid, 5 images per row.
    
    Args:
    - images (ndarray): An array of shape [N, 1, W, H]
    """
    N, _, W, H = images.shape

    # Determine the number of rows. If N is not a multiple of 5, we'll have some empty spots.
    num_rows = int(np.ceil(N / 5))

    # Set up the figure size
    fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
    
    for i in range(num_rows):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                # If there's an image left to display
                if num_rows == 1:
                    ax = axes[j] if N > 1 else axes
                else:
                    ax = axes[i, j]
                ax.imshow(images[idx, 0], cmap='gray')
                ax.axis('off')
            else:
                # Turn off axes for empty subplots
                if num_rows > 1:
                    axes[i, j].axis('off')
                else:
                    axes[j].axis('off')

    plt.tight_layout()
    plt.show()

import imageio

def create_gif_from_tensor(tensor: torch.Tensor, filename: str, duration: int = 100, interval: int = 1):
    """
    Create a gif from the given tensor and save all samples in one gif.

    Parameters:
    - tensor (torch.Tensor): A tensor of shape [N, T, C, W, H]
    - filename (str): Name of the gif file to save.
    - duration (int): Duration for each frame in the gif.
    - interval (int): Interval for sampling frames.

    Returns:
    - None. The gif is saved to the given filename.
    """

    # Check tensor dimensions
    if len(tensor.shape) != 5:
        raise ValueError("Expected tensor of shape [N, T, C, W, H], but got {}".format(tensor.shape))

    # Convert tensor to numpy array
    array = tensor.cpu().numpy()

    # Limit to the first 25 samples
    array = array[:25]

    all_frames = []

    max_frames = array.shape[1] // interval  # max number of frames after sampling

    for idx in range(max_frames):
        # For each frame, construct a mosaic of images
        mosaic_frame = []
        for row in range(5):  # 5 rows
            row_images = []
            for col in range(5):  # 5 columns
                n = row * 5 + col
                if n < array.shape[0]:
                    t = array.shape[1] - 1 - idx * interval
                    img = array[n, t]
                    # Normalize to [0, 255]
                    img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
                    if img.shape[0] == 1:  # Grayscale image
                        img = img.squeeze(0)
                    elif img.shape[0] == 3:  # RGB image
                        img = np.transpose(img, (1, 2, 0))
                    else:
                        raise ValueError("Unsupported number of channels: {}".format(img.shape[0]))
                    row_images.append(img)
                else:
                    # if no more data, append a blank image
                    if array.shape[2] == 3:
                        blank_img = np.zeros((array.shape[3], array.shape[4], 3), dtype=np.uint8)
                    else:
                        blank_img = np.zeros((array.shape[3], array.shape[4]), dtype=np.uint8)
                    row_images.append(blank_img)

            mosaic_frame.append(np.hstack(row_images))
        all_frames.append(np.vstack(mosaic_frame))

    # Save gif
    imageio.mimsave(filename + ".gif", all_frames, duration=duration)