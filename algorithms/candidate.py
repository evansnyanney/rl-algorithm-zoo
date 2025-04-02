import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gym
import random
import os
from tqdm import tqdm
import wandb

# Set seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -------------------------------
# Custom Decision Transformer Implementation
# -------------------------------
class TransformerBlock(nn.Module):
    """Simple Transformer block with multi-head attention and feedforward network."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.ln1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.ln2(x)
        return x

class SimpleDecisionTransformer(nn.Module):
    """Custom implementation of Decision Transformer with simplified architecture."""
    def __init__(self, state_dim, act_dim, hidden_dim=128, num_layers=3, num_heads=4, 
                 max_ep_len=100, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.max_ep_len = max_ep_len
        
        # Embeddings for states, actions, returns, and timesteps
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(act_dim, hidden_dim)
        self.return_encoder = nn.Linear(1, hidden_dim)
        self.timestep_embeddings = nn.Embedding(max_ep_len, hidden_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, 4 * hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head for action prediction
        self.action_predictor = nn.Linear(hidden_dim, act_dim)
        
    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq = states.shape[0], states.shape[1]
        
        # Encode inputs
        state_embeddings = self.state_encoder(states)
        action_embeddings = self.action_encoder(actions)
        return_embeddings = self.return_encoder(returns_to_go)
        time_embeddings = self.timestep_embeddings(timesteps)
        
        # Prepare transformer inputs by interleaving (r_t, s_t, a_t) for each timestep.
        transformer_inputs = torch.zeros((batch_size, seq * 3, self.hidden_dim), device=states.device)
        for i in range(seq):
            transformer_inputs[:, i * 3 + 0] = return_embeddings[:, i] + time_embeddings[:, i]
            transformer_inputs[:, i * 3 + 1] = state_embeddings[:, i] + time_embeddings[:, i]
            transformer_inputs[:, i * 3 + 2] = action_embeddings[:, i] + time_embeddings[:, i]
            
        # Process with transformer blocks
        x = transformer_inputs
        for block in self.transformer_blocks:
            x = block(x)
        
        # Extract action predictions (located after each state: index i*3+1)
        action_preds = torch.zeros((batch_size, seq, self.act_dim), device=states.device)
        for i in range(seq):
            action_preds[:, i] = self.action_predictor(x[:, i * 3 + 1])
            
        return action_preds

# -------------------------------
# Offline Dataset Preparation
# -------------------------------
def create_expert_policy_dataset(env, num_episodes=200, max_ep_len=500):
    """Generate a high-quality dataset using a pure expert policy for CartPole."""
    episodes = []
    
    def expert_policy(state):
        cart_pos, cart_vel, pole_angle, pole_vel = state
        if pole_vel > 0 and pole_angle > 0:
            return 1
        elif pole_vel < 0 and pole_angle < 0:
            return 0
        elif pole_angle > 0:
            return 1
        else:
            return 0
    
    for i in tqdm(range(num_episodes), desc="Collecting expert demonstrations"):
        episode = {"states": [], "actions": [], "rewards": []}
        state, _ = env.reset()
        done = False
        t = 0
        while not done and t < max_ep_len:
            action = expert_policy(state)
            if random.random() < 0.1:
                action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)
            action_one_hot = np.zeros(env.action_space.n)
            action_one_hot[action] = 1.0
            episode["states"].append(state)
            episode["actions"].append(action_one_hot)
            episode["rewards"].append(reward)
            state = next_state
            t += 1
            done = done or truncated
        if t >= 50:
            episodes.append(episode)
    
    print(f"Collected {len(episodes)} expert demonstrations")
    lengths = [len(ep["states"]) for ep in episodes]
    print(f"Episode length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    return episodes

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns-to-go for a list of rewards."""
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0
    for i in reversed(range(len(rewards))):
        running_return = rewards[i] + gamma * running_return
        returns[i] = running_return
    return returns

class OfflineDataset(Dataset):
    """
    Offline dataset for Decision Transformer training.
    Each sample is a fixed-length segment (of timesteps) from an episode.
    """
    def __init__(self, episodes, state_dim, act_dim, seq_len=20, gamma=0.99):
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.samples = []
        
        for ep in episodes:
            states = np.array(ep["states"])
            actions = np.array(ep["actions"])
            rewards = np.array(ep["rewards"])
            returns = compute_returns(rewards, gamma)
            
            ep_len = len(states)
            for i in range(ep_len - 1):
                sample_len = min(seq_len, ep_len - i)
                state_segment = states[i:i + sample_len]
                action_segment = actions[i:i + sample_len]
                return_segment = returns[i:i + sample_len].reshape(-1, 1)
                if sample_len < seq_len:
                    state_padding = np.zeros((seq_len - sample_len, self.state_dim))
                    action_padding = np.zeros((seq_len - sample_len, self.act_dim))
                    return_padding = np.zeros((seq_len - sample_len, 1))
                    state_segment = np.vstack([state_segment, state_padding])
                    action_segment = np.vstack([action_segment, action_padding])
                    return_segment = np.vstack([return_segment, return_padding])
                timesteps = np.arange(seq_len)
                sample = {
                    "states": torch.FloatTensor(state_segment),
                    "actions": torch.FloatTensor(action_segment),
                    "returns": torch.FloatTensor(return_segment),
                    "timesteps": torch.LongTensor(timesteps),
                    "sample_len": sample_len
                }
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# -------------------------------
# Evaluation Helper for Training
# -------------------------------
def evaluate_decision_transformer_in_training(model, env_name="CartPole-v1", target_return=500, num_eval_episodes=5):
    """
    Run a quick evaluation over a few episodes using the trained model.
    Returns the average reward ("score").
    """
    device = next(model.parameters()).device
    scores = []
    
    for ep in range(num_eval_episodes):
        env = gym.make(env_name)
        state, _ = env.reset()
        done = False
        ep_reward = 0
        timestep = 0
        
        states_history = []
        actions_history = []
        returns_history = []
        
        while not done:
            states_history.append(state)
            remaining_return = max(0, target_return - ep_reward)
            returns_history.append([remaining_return])
            
            action = select_action_dt(model, states_history, actions_history, returns_history, timestep, remaining_return, device)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            ep_reward += reward
            state = next_state
            
            action_one_hot = np.zeros(env.action_space.n)
            action_one_hot[action] = 1.0
            actions_history.append(action_one_hot)
            
            timestep += 1
            if timestep >= 1000:
                break
        
        scores.append(ep_reward)
        env.close()
    return np.mean(scores)

# -------------------------------
# Training and Evaluation
# -------------------------------
def train_decision_transformer(env_name, num_episodes=100, seq_len=20, batch_size=64, 
                               num_epochs=10, gamma=0.99, lr=1e-4, save_model=True, 
                               log_wandb=True, project="rl_algorithm_zoo"):
    """Train a Decision Transformer on offline data."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {act_dim}")
    print(f"Sequence length (timesteps): {seq_len}")
    
    if log_wandb:
        wandb.init(
            project=project, 
            group="rl_algorithm_zoo",
            name="DecisionTransformerTraining", 
            config={
                "env_name": env_name,
                "num_episodes": num_episodes,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "gamma": gamma,
                "lr": lr,
                "hidden_dim": 256,
                "num_layers": 4,
                "num_heads": 8,
                "dropout": 0.1
            }
        )
    
    episodes = create_expert_policy_dataset(env, num_episodes, max_ep_len=500)
    dataset = OfflineDataset(episodes, state_dim, act_dim, seq_len, gamma)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=256,
        max_ep_len=seq_len,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    epoch_losses = []
    best_loss = float('inf')
    eval_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        epoch_progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in epoch_progress:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            returns = batch["returns"].to(device)
            timesteps = batch["timesteps"].to(device)
            sample_lens = batch["sample_len"]
            
            action_preds = model(states, actions, returns, timesteps)
            
            loss = 0
            b_size = states.shape[0]
            for i in range(b_size):
                valid_len = sample_lens[i]
                if valid_len > 1:
                    pred = action_preds[i, :valid_len - 1]
                    target = actions[i, 1:valid_len]
                    loss += F.mse_loss(pred, target)
            loss = loss / b_size
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            epoch_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / total_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss and save_model:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_decision_transformer.pt")
            print(f"Saved best model with loss: {best_loss:.6f}")
        
        # Evaluate model after each epoch to obtain a "score"
        model.eval()
        avg_score = evaluate_decision_transformer_in_training(model, env_name=env_name, target_return=500, num_eval_episodes=5)
        eval_scores.append(avg_score)
        
        if log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "best_loss": best_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "score": avg_score,
                "episode": epoch + 1
            })
    
    if save_model and os.path.exists("best_decision_transformer.pt"):
        model.load_state_dict(torch.load("best_decision_transformer.pt"))
        print("Loaded best model for evaluation")
    
    if log_wandb:
        wandb.finish()
    
    return {"model": model, "env_info": {"state_dim": state_dim, "act_dim": act_dim},
            "epoch_losses": epoch_losses, "eval_scores": eval_scores}

def plot_training_results(results):
    """Plot training loss and evaluation scores for the Decision Transformer."""
    import matplotlib.pyplot as plt
    epoch_losses = results.get("epoch_losses", [])
    eval_scores = results.get("eval_scores", [])
    if not epoch_losses or not eval_scores:
        print("No training data to plot.")
        return
    
    # Create a figure with 2 subplots sharing the x-axis (epochs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    ax1.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o', label="Offline Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Decision Transformer Offline Training Loss")
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(range(1, len(eval_scores)+1), eval_scores, marker='x', color="tab:blue", label="Evaluation Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Decision Transformer Evaluation Score")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("decision_transformer_training_results.png")
    plt.show()

# -------------------------------
# Action Selection and Final Evaluation
# -------------------------------
def select_action_dt(model, states_history, actions_history, returns_history, timestep, target_return, device):
    """
    Select action using a rolling window approach with the Decision Transformer model.
    This version pads the histories so that states, actions, and returns all have length equal
    to model.max_ep_len.
    """
    max_seq = model.max_ep_len
    state_dim = model.state_dim
    act_dim = model.act_dim

    # Convert histories to tensors
    if len(states_history) > 0:
        states = torch.FloatTensor(np.array([np.asarray(s) for s in states_history])).to(device)
    else:
        states = torch.zeros((0, state_dim), device=device)
    
    if len(actions_history) > 0:
        actions = torch.FloatTensor(np.array([np.asarray(a) for a in actions_history])).to(device)
    else:
        actions = torch.zeros((0, act_dim), device=device)
    
    if len(returns_history) > 0:
        returns = torch.FloatTensor(np.array(returns_history)).to(device)
    else:
        returns = torch.zeros((0, 1), device=device)
    
    # Ensure actions and returns have the same number of rows as states
    current_len = states.shape[0]
    if actions.shape[0] < current_len:
        pad_count = current_len - actions.shape[0]
        pad_actions = torch.zeros((pad_count, act_dim), device=device)
        actions = torch.cat([pad_actions, actions], dim=0)
    if returns.shape[0] < current_len:
        pad_count = current_len - returns.shape[0]
        pad_returns = torch.zeros((pad_count, 1), device=device)
        returns = torch.cat([pad_returns, returns], dim=0)
    
    # Now adjust the histories to have exactly max_seq rows
    if current_len > max_seq:
        states = states[-max_seq:]
        actions = actions[-max_seq:]
        returns = returns[-max_seq:]
    elif current_len < max_seq:
        pad = max_seq - current_len
        pad_states = torch.zeros((pad, state_dim), device=device)
        pad_actions = torch.zeros((pad, act_dim), device=device)
        pad_returns = torch.zeros((pad, 1), device=device)
        states = torch.cat([pad_states, states], dim=0)
        actions = torch.cat([pad_actions, actions], dim=0)
        returns = torch.cat([pad_returns, returns], dim=0)
    
    # Reshape to (1, max_seq, *)
    states = states.unsqueeze(0)
    actions = actions.unsqueeze(0)
    returns = returns.unsqueeze(0)
    timesteps = torch.arange(max_seq, device=device).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        action_preds = model(states, actions, returns, timesteps)
        action = torch.argmax(action_preds[0, -1]).item()
    return action

def evaluate_decision_transformer(model, env_name="CartPole-v1", target_return=500, num_episodes=5, render=False):
    """Evaluate the trained Decision Transformer using a rolling window approach."""
    env = gym.make(env_name, render_mode="human" if render else None)
    device = next(model.parameters()).device
    scores = []
    
    print(f"Evaluation config: target_return={target_return}, num_episodes={num_episodes}")
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        timestep = 0
        
        states_history = []
        actions_history = []
        returns_history = []
        
        while not done:
            states_history.append(state)
            remaining_return = max(0, target_return - ep_reward)
            returns_history.append([remaining_return])
            
            action = select_action_dt(model, states_history, actions_history, returns_history, timestep, remaining_return, device)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            ep_reward += reward
            state = next_state
            
            action_one_hot = np.zeros(env.action_space.n)
            action_one_hot[action] = 1.0
            actions_history.append(action_one_hot)
            
            timestep += 1
            if timestep >= 1000:
                break
        
        scores.append(ep_reward)
        print(f"Episode {ep+1}: Score: {ep_reward}")
    
    env.close()
    avg_score = np.mean(scores)
    print(f"Average Score over {num_episodes} episodes: {avg_score}")
    return scores

# -------------------------------
# Main Function
# -------------------------------
if __name__ == "__main__":
    SEQ_LEN = 20  # Use 20 timesteps as specified
    print(f"Training with sequence length (timesteps): {SEQ_LEN}")
    
    results = train_decision_transformer(
        env_name="CartPole-v1",
        num_episodes=50,      # Fewer episodes for quicker runs
        seq_len=SEQ_LEN,
        batch_size=64,
        num_epochs=10,        # More epochs for better convergence
        gamma=0.99,
        save_model=True,
        log_wandb=True
    )
    
    print("Training completed successfully!")
    
    try:
        model = results["model"]
        scores = evaluate_decision_transformer(
            model,
            env_name="CartPole-v1",
            target_return=500,
            num_episodes=100,
            render=False
        )
        print(f"Evaluation completed with scores: {scores}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        
    plot_training_results(results)
    print("Script execution completed.")
