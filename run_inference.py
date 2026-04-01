import argparse
import importlib
import sys
from pathlib import Path
from typing import List

def get_available_models() -> List[str]:
    """Discover available models based on directory names in 'models'."""
    repo_root = Path(__file__).parent
    models_dir = repo_root / "models"
    models = []
    if models_dir.exists() and models_dir.is_dir():
        for d in models_dir.iterdir():
            if d.is_dir() and (d / "shared_runner.py").exists() and not d.name.startswith("_"):
                models.append(d.name)
    return sorted(models)

def main() -> None:
    available_models = get_available_models()
    
    parser = argparse.ArgumentParser(description="Unified runner for all audio evaluation models.")
    parser.add_argument("--model", type=str, required=True, choices=available_models, help="Which model to run")
    parser.add_argument("--task", type=str, help="Run a specific task id")
    parser.add_argument("--all", action="store_true", help="Run all tasks for the given model")
    
    # Optional overrides, if provided they will override the model's defaults
    parser.add_argument("--max-workers", type=int, default=None, help="Override default max workers")
    parser.add_argument("--save-interval", type=int, default=None, help="Override default save interval")
    parser.add_argument("--model-name", type=str, default=None, help="Override default model name string")
    
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("Please provide --task or --all")

    # Dynamically import the model's runner and configurations
    try:
        shared_runner = importlib.import_module(f"models.{args.model}.shared_runner")
        task_configs = importlib.import_module(f"models.{args.model}.task_configs")
    except ImportError as e:
        print(f"Error importing modules for model {args.model}: {e}")
        sys.exit(1)

    # Validate task selection
    available_tasks = getattr(task_configs, "TASK_IDS", [])
    if not available_tasks:
        print(f"Error: No TASK_IDS found in models.{args.model}.task_configs")
        sys.exit(1)

    if args.task and args.task not in available_tasks:
        print(f"Error: Task '{args.task}' is not available for model {args.model}.")
        print(f"Available tasks for {args.model}: {', '.join(available_tasks)}")
        sys.exit(1)

    task_ids = available_tasks if args.all else [args.task]

    # Resolve default arguments from the chosen model's runner
    max_workers = args.max_workers if args.max_workers is not None else getattr(shared_runner, "MAX_WORKERS_DEFAULT", 4)
    save_interval = args.save_interval if args.save_interval is not None else getattr(shared_runner, "SAVE_INTERVAL_DEFAULT", 10)
    model_name = args.model_name if args.model_name is not None else getattr(shared_runner, "MODEL_NAME_DEFAULT", args.model)

    print(f"Starting evaluation for model: {args.model}")
    print(f"Model internal name: {model_name}, Max workers: {max_workers}, Save interval: {save_interval}")

    overall = {"total": 0, "processed": 0, "done_before": 0, "errors": 0}
    for task_id in task_ids:
        stats = shared_runner.run_task(
            task_id=task_id,
            max_workers=max_workers,
            save_interval=save_interval,
            model_name=model_name,
        )
        for key in overall:
            overall[key] += stats.get(key, 0)
            
    print("\nOverall:", overall)

if __name__ == "__main__":
    main()
