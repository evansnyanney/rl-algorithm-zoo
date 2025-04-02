import argparse
import wandb

# Log in to WandB if logging is enabled.
# (For security, consider using an environment variable instead of hardcoding the API key.)
def wandb_login_if_enabled(enable_logging):
    if enable_logging:
        wandb.login(key="911b0b7fc05751caed5698362eafac68d34da465")

def main():
    parser = argparse.ArgumentParser(description="RL Algorithm Zoo - Training Script")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["dqn", "reinforce", "a2c", "decision_transformer"],
        help="Select the RL algorithm to train"
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--log_wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()

    # Log in to WandB if enabled
    wandb_login_if_enabled(args.log_wandb)

    if args.algo == "dqn":
        from algorithms.value_based import train_dqn, plot_training_results
        results = train_dqn(
            env_name=args.env,
            num_episodes=args.episodes,
            log_wandb=args.log_wandb,
            render=args.render
        )
        plot_training_results(results)
    elif args.algo == "reinforce":
        from algorithms.policy_based import train_reinforce, plot_training_results
        results = train_reinforce(
            env_name=args.env,
            num_episodes=args.episodes,
            log_wandb=args.log_wandb,
            render=args.render
        )
        plot_training_results(results)
    elif args.algo == "a2c":
        from algorithms.hybrid import train_a2c, plot_training_results
        results = train_a2c(
            env_name=args.env,
            num_episodes=args.episodes,
            log_wandb=args.log_wandb,
            render=args.render
        )
        plot_training_results(results)
    elif args.algo == "decision_transformer":
        from algorithms.candidate import train_decision_transformer
        results = train_decision_transformer(
            env_name=args.env,
            num_episodes=args.episodes,
            log_wandb=args.log_wandb,
            project="rl_algorithm_zoo"
        )
        print("Training completed for Decision Transformer. Check your wandb dashboard for results.")

if __name__ == "__main__":
    main()
