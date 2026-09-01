import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        prog="hrl_extreme",
        description="HRL Project Extreme: Hierarchical Reinforcement Learning Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Web Dashboard command
    web_parser = subparsers.add_parser("web", help="Start the interactive web visualizer dashboard")
    web_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to bind")
    web_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run policy training")
    train_parser.add_argument("--mode", type=str, default="continuous", choices=["continuous", "discrete"])
    train_parser.add_argument("--episodes", type=int, default=100)
    train_parser.add_argument("--vec-envs", type=int, default=4)
    train_parser.add_argument("--c-step", type=int, default=8)
    train_parser.add_argument("--no-torch", action="store_true")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run trajectory evaluation")
    eval_parser.add_argument("--mode", type=str, default="continuous", choices=["continuous", "discrete"])
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--c-step", type=int, default=8)
    eval_parser.add_argument("--no-torch", action="store_true")

    args = parser.parse_args()

    if args.command == "web" or args.command is None:
        from .server import run_server
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8000)
        run_server(host=host, port=port)
    elif args.command == "train":
        from .train import train_hrl
        train_hrl(
            mode=args.mode,
            num_episodes=args.episodes,
            vec_envs=args.vec_envs,
            c_step=args.c_step,
            use_torch=not args.no_torch,
        )
    elif args.command == "evaluate":
        from .evaluate import evaluate_agent
        evaluate_agent(
            mode=args.mode,
            episodes=args.episodes,
            c_step=args.c_step,
            use_torch=not args.no_torch,
        )

if __name__ == "__main__":
    main()
