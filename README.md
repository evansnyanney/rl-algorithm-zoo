# rl-algorithm-zoo
A comparative study of reinforcement learning algorithms on CartPole-v1, including REINFORCE, DQN, A2C, and Decision Transformer


# RL Algorithm Zoo: A Comparative Study on CartPole-v1

[![WandB Project](https://wandb.ai/evansnyanney-ohio-university/rl_algorithm_zoo/reports/Comparison-of-Four-Reinforcement-Learning-Methods-on-CartPole-v1--VmlldzoxMjEwMzE3MQ)](https://wandb.ai/evansnyanney/rl_algorithm-zoo)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/evansnyanney/rl-algorithm-zoo/main/LICENSE)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![GitHub Repo Size](https://img.shields.io/github/repo-size/evansnyanney/rl-algorithm-zoo)
![GitHub Stars](https://img.shields.io/github/stars/evansnyanney/rl-algorithm-zoo?style=social)
![GitHub Forks](https://img.shields.io/github/forks/evansnyanney/rl-algorithm-zoo?style=social)

**RL Algorithm Zoo** contains implementations of four reinforcement learning methods applied to the CartPole-v1 task. The project compares a policy-based method (REINFORCE), a value-based method (DQN), a hybrid method (A2C), and an offline candidate (Decision Transformer).

---

## Table of Contents

- [Overview](#overview)
- [Key Performance Metrics](#key-performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
  - [REINFORCE](#reinforce-policy-based)
  - [DQN (Value-Based)](#dqn-value-based)
  - [A2C (Hybrid)](#a2c-hybrid)
  - [Decision Transformer (Candidate)](#decision-transformer-candidate)
- [Experiment Tracking](#experiment-tracking)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project compares four RL algorithms on CartPole-v1 using two primary metrics: average return and average episode length (which are equivalent in CartPole, since each timestep gives a reward of 1). The methods examined are:

- **REINFORCE (Policy-Based):** A classic policy-gradient method.
- **DQN (Value-Based):** A deep Q-network for estimating action values.
- **A2C (Hybrid):** An actor-critic approach that combines policy and value learning.
- **Decision Transformer (Candidate):** An offline method that learns from expert demonstrations.

---

## Key Performance Metrics

The following table summarizes the final evaluation metrics for each algorithm:

| **Algorithm**                   | **Average Return** | **Average Episode Length** |
|---------------------------------|--------------------|----------------------------|
| **REINFORCE (Policy-Based)**    | 500.00             | 500                        |
| **DQN (Value-Based)**           | 9.40               | 9                          |
| **A2C (Hybrid)**                | 9.63               | 9–10                       |
| **Decision Transformer (Candidate)** | 10.40        | ~10                        |

*Note: In CartPole-v1 the reward per timestep equals the episode length.*

---

## Installation

Clone the repository and set up your environment:

```sh
git clone https://github.com/evansnyanney/rl-algorithm-zoo.git
cd rl-algorithm-zoo
python -m venv .venv
