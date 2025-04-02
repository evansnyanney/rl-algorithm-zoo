import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import gym
import wandb
import matplotlib.pyplot as plt

# Defining the Actor network
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    
    def act(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

# Defining the Critic network
class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Defining the A2C agent
class A2CAgent:
    def __init__(self, state_size, action_size, hidden_size=128, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, entropy_coef=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.actor = ActorNetwork(state_size, action_size, hidden_size)
        self.critic = CriticNetwork(state_size, hidden_size)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def step(self, state):
        action, log_prob = self.actor.act(state)
        return action, log_prob
    
    def update(self, state, reward, next_state, done, log_prob):
        # Converting states to tensors
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
        
        # Computing the state value and next state value
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)
        
        # Computing the TD target and advantage
        td_target = reward + self.gamma * next_value * (1 - int(done))
        advantage = td_target - value
        
        # Actor loss: negative log-prob multiplied by advantage plus an entropy bonus for exploration
        actor_loss = -log_prob * advantage.detach()
        probs = self.actor(state_tensor)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        actor_loss -= self.entropy_coef * entropy
        
        # Critic loss: mean squared error between predicted value and TD target
        critic_loss = F.mse_loss(value, td_target.detach())
        
        # Updating actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Updating the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

def train_a2c(env_name="CartPole-v1", num_episodes=1000, max_steps=1000, 
               log_wandb=True, render=False, render_every=100):
    """Train an A2C agent on a gym environment."""
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = A2CAgent(state_size, action_size)
    
    if log_wandb:
        wandb.init(project="rl_algorithm_zoo", name="A2C", config={
            "env_name": env_name,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "state_size": state_size,
            "action_size": action_size,
            "hidden_size": 128,
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "entropy_coef": 0.01
        })
    
    scores = []
    actor_losses = []
    critic_losses = []
    episode_lengths = []
    running_reward = 10
    
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for t in range(max_steps):
            if render and episode % render_every == 0:
                env.render()
            
            action, log_prob = agent.step(state)
            next_state, reward, done, truncated, _ = env.step(action)
            terminated = done or truncated
            episode_reward += reward
            
            a_loss, c_loss = agent.update(state, reward, next_state, terminated, log_prob)
            actor_losses.append(a_loss)
            critic_losses.append(c_loss)
            
            state = next_state
            steps = t + 1
            if terminated:
                break
        
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
                "actor_loss": np.mean(actor_losses[-10:]) if len(actor_losses) >= 10 else np.mean(actor_losses),
                "critic_loss": np.mean(critic_losses[-10:]) if len(critic_losses) >= 10 else np.mean(critic_losses),
                "avg_score_100": np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            })
        
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is {running_reward:.2f}, above threshold {env.spec.reward_threshold}")
            break
    
    env.close()
    
    if log_wandb:
        wandb.finish()
    
    return {
        "scores": scores,
        "episode_lengths": episode_lengths,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "agent": agent
    }

def plot_training_results(results):
    """Plot training scores, episode lengths, and losses."""
    scores = results["scores"]
    episode_lengths = results["episode_lengths"]
    actor_losses = results["actor_losses"]
    critic_losses = results["critic_losses"]
    
    window_size = min(100, len(scores))
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot scores
    ax1.plot(np.arange(len(scores)), scores, label='Score', alpha=0.6)
    ax1.plot(np.arange(window_size-1, len(scores)), moving_avg, label=f'{window_size}-episode Moving Avg', color='red')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('A2C Training Scores')
    ax1.legend()
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(np.arange(len(episode_lengths)), episode_lengths, label='Episode Length', color='purple')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Length Over Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot actor loss
    ax3.plot(np.arange(len(actor_losses)), actor_losses, label='Actor Loss')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Actor Loss')
    ax3.set_title('Actor Loss')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)
    
    # Plot critic loss
    ax4.plot(np.arange(len(critic_losses)), critic_losses, label='Critic Loss')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Critic Loss')
    ax4.set_title('Critic Loss')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('a2c_training_results.png')
    plt.show()

def evaluate_a2c(agent, env_name="CartPole-v1", num_episodes=10, render=True):
    """Evaluate a trained A2C agent."""
    env = gym.make(env_name, render_mode='human' if render else None)
    scores = []
    
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            if render:
                env.render()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs = agent.actor(state_tensor)
            action = torch.argmax(probs, dim=1).item()
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
    # Train the A2C agent
    results = train_a2c(env_name="CartPole-v1", num_episodes=1000, log_wandb=False)
    
    # Plot the training results
    plot_training_results(results)
    
    # Evaluate the trained agent
    evaluate_a2c(results["agent"], env_name="CartPole-v1", num_episodes=5, render=True)
