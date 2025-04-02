import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gym
import random
from collections import deque, namedtuple
import wandb

# Define the DQN model architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3, 
                 gamma=0.99, buffer_size=10000, batch_size=64, 
                 update_every=4, tau=1e-3, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Initialize a DQN Agent
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Number of possible actions
            hidden_size (int): Size of hidden layers
            lr (float): Learning rate
            gamma (float): Discount factor
            buffer_size (int): Size of replay buffer
            batch_size (int): Mini-batch size
            update_every (int): How often to update the network
            tau (float): Soft update parameter
            eps_start (float): Starting value of epsilon for epsilon-greedy action selection
            eps_end (float): Minimum value of epsilon
            eps_decay (float): Decay rate for epsilon
        """
        # Initialize parameters
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.tau = tau
        
        # Q-Network
        self.qnetwork_local = DQN(state_size, action_size, hidden_size)
        self.qnetwork_target = DQN(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize epsilon for exploration
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn from experiences if enough samples are available
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def act(self, state, eps=None):
        """Returns action for given state following epsilon-greedy policy"""
        if eps is None:
            eps = self.eps
            
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences
        
        # Get max Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        return loss.item()
    
    def soft_update(self, local_model, target_model):
        """Soft update target network parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def update_epsilon(self):
        """Decay epsilon for epsilon-greedy policy"""
        self.eps = max(self.eps_end, self.eps_decay * self.eps)

def train_dqn(env_name="CartPole-v1", num_episodes=500, max_steps=1000, 
              log_wandb=True, render=False, render_every=100):
    """Train a DQN agent on the specified gym environment"""
    
    # Initialize environment
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Initialize WandB logging
    if log_wandb:
        wandb.init(project="rl_algorithm_zoo", name="DQN", config={
            "env_name": env_name,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "state_size": state_size,
            "action_size": action_size,
            "hidden_size": 64,
            "lr": 1e-3,
            "gamma": 0.99,
            "buffer_size": 10000,
            "batch_size": 64,
            "update_every": 4,
            "tau": 1e-3,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "eps_decay": 0.995
        })
    
    # Training loop
    scores = []
    losses = []
    eps_history = []
    episode_lengths = []
    
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        steps = 0
        
        for t in range(max_steps):
            # Render environment (optional)
            if render and episode % render_every == 0:
                env.render()
            
            # Select and perform action
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            terminated = done or truncated
            
            # Store transition and learn
            agent.step(state, action, reward, next_state, terminated)
            
            # Update state and score
            state = next_state
            score += reward
            steps = t + 1
            
            if terminated:
                break
                
        # Update epsilon
        agent.update_epsilon()
        
        # Store metrics
        scores.append(score)
        episode_lengths.append(steps)
        eps_history.append(agent.eps)
        
        # Print progress
        print(f"Episode {episode}/{num_episodes} | Score: {score:.2f} | Epsilon: {agent.eps:.4f} | Episode Length: {steps}")
        
        # Log to WandB
        if log_wandb:
            wandb.log({
                "episode": episode,
                "score": score,
                "episode_length": steps,
                "epsilon": agent.eps,
                "avg_score_100": np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            })
    
    # Close environment
    env.close()
    
    if log_wandb:
        wandb.finish()
    
    # Return training results
    return {
        "scores": scores,
        "eps_history": eps_history,
        "episode_lengths": episode_lengths,
        "agent": agent
    }

def plot_training_results(results):
    """Plot training results for DQN."""
    scores = results["scores"]
    eps_history = results["eps_history"]
    episode_lengths = results["episode_lengths"]
    
    window_size = min(100, len(scores))
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    
    # Create figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    
    # Plot scores
    axs[0].plot(np.arange(len(scores)), scores, label='Score', alpha=0.6)
    axs[0].plot(np.arange(window_size-1, len(scores)), moving_avg, label=f'{window_size}-episode Moving Avg', color='red')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Score')
    axs[0].set_title('DQN Training Scores')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot epsilon decay
    axs[1].plot(np.arange(len(eps_history)), eps_history)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Epsilon')
    axs[1].set_title('Epsilon Decay')
    axs[1].grid(True)
    
    # Plot episode lengths
    axs[2].plot(np.arange(len(episode_lengths)), episode_lengths, label='Episode Length', color='purple')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Episode Length')
    axs[2].set_title('Episode Length Over Time')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_training_results.png')
    plt.show()

def evaluate_dqn(agent, env_name="CartPole-v1", num_episodes=10, render=True):
    """Evaluate a trained DQN agent"""
    env = gym.make(env_name, render_mode='human' if render else None)
    scores = []
    
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # Use greedy policy (epsilon=0)
            action = agent.act(state, eps=0)
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
    # Train DQN agent
    results = train_dqn(env_name="CartPole-v1", num_episodes=500, log_wandb=False)
    
    # Plot training results
    plot_training_results(results)
    
    # Evaluate the trained agent
    agent = results["agent"]
    evaluate_dqn(agent, env_name="CartPole-v1", num_episodes=5, render=True)
