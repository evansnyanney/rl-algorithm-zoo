import wandb

def init_wandb(project_name, run_name, config):
    """Initialize a WandB run."""
    wandb.init(project=project_name, name=run_name, config=config)

def log_metrics(metrics, step=None):
    """Log a dictionary of metrics."""
    wandb.log(metrics, step=step)

def finish_wandb():
    """Finish the WandB run."""
    wandb.finish()
