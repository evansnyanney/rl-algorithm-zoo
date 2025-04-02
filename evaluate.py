import argparse

def main():
    parser = argparse.ArgumentParser(description="RL Algorithm Zoo - Evaluation Script")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["dqn", "reinforce", "a2c", "decision_transformer"],
        help="Select the RL algorithm to evaluate"
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation")
    parser.add_argument("--target_return", type=float, default=50, help="Target return for Decision Transformer")
    # This argument is reserved for future model loading functionality.
    parser.add_argument("--model_path", type=str, default=None, help="Path to load a saved model (not implemented)")
    # If no saved model is provided, we train a small model for evaluation.
    parser.add_argument("--train_episodes", type=int, default=50, help="Episodes to train if no model is loaded")
    args = parser.parse_args()

    if args.algo == "dqn":
        from algorithms.value_based import train_dqn, evaluate_dqn
        if args.model_path is None:
            print("No saved model provided. Training DQN agent for evaluation.")
            results = train_dqn(env_name=args.env, num_episodes=args.train_episodes, log_wandb=False, render=False)
            agent = results["agent"]
        else:
            # Model loading not implemented—this is a placeholder.
            agent = None
        evaluate_dqn(agent, env_name=args.env, num_episodes=args.eval_episodes, render=args.render)

    elif args.algo == "reinforce":
        from algorithms.policy_based import train_reinforce, evaluate_reinforce
        if args.model_path is None:
            print("No saved model provided. Training REINFORCE agent for evaluation.")
            results = train_reinforce(env_name=args.env, num_episodes=args.train_episodes, log_wandb=False, render=False)
            agent = results["agent"]
        else:
            agent = None
        evaluate_reinforce(agent, env_name=args.env, num_episodes=args.eval_episodes, render=args.render)

    elif args.algo == "a2c":
        from algorithms.hybrid import train_a2c, evaluate_a2c
        if args.model_path is None:
            print("No saved model provided. Training A2C agent for evaluation.")
            results = train_a2c(env_name=args.env, num_episodes=args.train_episodes, log_wandb=False, render=False)
            agent = results["agent"]
        else:
            agent = None
        evaluate_a2c(agent, env_name=args.env, num_episodes=args.eval_episodes, render=args.render)

    elif args.algo == "decision_transformer":
        from algorithms.candidate import train_decision_transformer, evaluate_decision_transformer
        if args.model_path is None:
            print("No saved model provided. Training Decision Transformer for evaluation.")
            results = train_decision_transformer(env_name=args.env, num_episodes=args.train_episodes, log_wandb=False)
            model = results["model"]
        else:
            model = None
        evaluate_decision_transformer(model, env_name=args.env, target_return=args.target_return, num_episodes=args.eval_episodes, render=args.render)

if __name__ == "__main__":
    main()
