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
