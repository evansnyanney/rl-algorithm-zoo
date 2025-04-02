import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores, window=100, title='Training Scores', save_path='scores.png'):
    """Plot scores and a moving average over episodes."""
    moving_avg = np.convolve(scores, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Score', alpha=0.6)
    plt.plot(np.arange(window - 1, len(scores)), moving_avg, label=f'{window}-episode Moving Avg', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def plot_losses(losses, title='Training Loss', save_path='losses.png'):
    """Plot training loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Episode or Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
