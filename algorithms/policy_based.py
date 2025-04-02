import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym
import wandb

# Defining the policy network (actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        # Converting the state from numpy to torch tensor, adding batch dimension.
        state = torch.from_numpy(state).float().unsqueeze(0)
        # Do NOT detach so gradients can flow through this computation.
        probs = self.forward(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

class REINFORCEAgent:
    def __init__(self, state_size, action_size, hidden_size=128, lr=0.01, gamma=0.99):
        """Initialize a REINFORCE Agent.
        
        Args:
            state_size (int): Dimension of each state.
            action_size (int): Number of possible actions.
            hidden_size (int): Size of the hidden layer.
            lr (float): Learning rate.
            gamma (float): Discount factor.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Policy network
        self.policy = PolicyNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for episode data
        self.saved_log_probs = []
        self.rewards = []
    
    def step(self, state):
        """Select an action based on the current policy."""
        action, log_prob = self.policy.act(state)
        self.saved_log_probs.append(log_prob)
        return action
    
    def add_reward(self, reward):
        """Store a reward."""
        self.rewards.append(reward)
    
    def learn(self):
        """Update policy parameters using the collected rewards and log probabilities."""
        returns = self._compute_returns(self.rewards, self.gamma)
        returns = torch.tensor(returns)
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate the loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Perform backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def _compute_returns(self, rewards, gamma):
        """Compute discounted returns for each timestep."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

def train_reinforce(env_name="CartPole-v1", num_episodes=1000, max_steps=1000, 
                   log_wandb=True, render=False, render_every=100):
    """Train a REINFORCE agent on the specified gym environment."""
    
    # Initializing the environment
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent
    agent = REINFORCEAgent(state_size=state_size, action_size=action_size)
    
    # Initialize WandB logging if desired
    if log_wandb:
        wandb.init(project="rl_algorithm_zoo", name="REINFORCE", config={
            "env_name": env_name,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "state_size": state_size,
            "action_size": action_size,
            "hidden_size": 128,
            "lr": 0.01,
            "gamma": 0.99
        })
    
    scores = []
    losses = []
    episode_lengths = []
    running_reward = 10
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for t in range(max_steps):
            # Render environment if desired
            if render and episode % render_every == 0:
                env.render()
            
            # Select and perform action
            action = agent.step(state)
            next_state, reward, done, truncated, _ = env.step(action)
            terminated = done or truncated
            
            # Store reward
            agent.add_reward(reward)
            episode_reward += reward
            
            # Update state
            state = next_state
            steps = t + 1
            if terminated:
                break
        
        # Update policy at the end of the episode
        loss = agent.learn()
        losses.append(loss)
        scores.append(episode_reward)
        episode_lengths.append(steps)
        
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} | Score: {episode_reward:.2f} | Running reward: {running_reward:.2f} | Episode length: {steps}")
        
        if log_wandb:
            wandb.log({
                "episode": episode,
                "score": episode_reward,
                "running_reward": running_reward,
                "episode_length": steps,
                "loss": loss,
                "avg_score_100": np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            })
        
        # Check if solved (if environment defines a reward threshold)
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is {running_reward:.2f}, above threshold {env.spec.reward_threshold}!")
            break
    
    env.close()
    
    if log_wandb:
        wandb.finish()
    
    return {"scores": scores, "losses": losses, "episode_lengths": episode_lengths, "agent": agent}

def plot_training_results(results):
    """Plot training results."""
    scores = results["scores"]
    losses = results["losses"]
    episode_lengths = results["episode_lengths"]
    
    window_size = min(100, len(scores))
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    
    # Plot scores
    axs[0].plot(np.arange(len(scores)), scores, label='Score', alpha=0.6)
    axs[0].plot(np.arange(window_size-1, len(scores)), moving_avg, label=f'{window_size}-episode Moving Avg', color='red')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Score')
    axs[0].set_title('REINFORCE Training Scores')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot episode lengths
    axs[1].plot(np.arange(len(episode_lengths)), episode_lengths, label='Episode Length', color='purple')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Episode Length')
    axs[1].set_title('Episode Length Over Time')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot losses (using a log scale)
    axs[2].plot(np.arange(len(losses)), losses, label='Loss')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Loss')
    axs[2].set_title('Policy Loss')
    axs[2].set_yscale('log')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('reinforce_training_results.png')
    plt.show()

def evaluate_reinforce(agent, env_name="CartPole-v1", num_episodes=10, render=True):
    """Evaluate a trained REINFORCE agent."""
    env = gym.make(env_name, render_mode='human' if render else None)
    scores = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # Use the policy for action selection (no exploration here)
            action, _ = agent.policy.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            state = next_state
            score += reward
        
        scores.append(score)
        print(f"Evaluation Episode {episode}/{num_episodes} | Score: {score:.2f}")
    
    env.close()
    print(f"Average Score: {np.mean(scores):.2f}")
    return scores

if __name__ == "__main__":
    # Train the REINFORCE agent
    results = train_reinforce(env_name="CartPole-v1", num_episodes=1000, log_wandb=False)
    
    # Plot training results
    plot_training_results(results)
    
    # Evaluate the trained agent
    agent = results["agent"]
    evaluate_reinforce(agent, env_name="CartPole-v1", num_episodes=5, render=True)
