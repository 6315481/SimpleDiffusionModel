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