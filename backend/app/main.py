import argparse
import getpass
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, configure_logging, get_logger

logger = get_logger("backend_main")


def default_user() -> str:
    return os.environ.get("ATLAS_USER") or getpass.getuser()


def parse_arguments(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Atlas - development command line interface"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="Write logs here instead of the default"
    )
    parser.add_argument(
        "--log-json", action="store_true", help="Emit logs as one JSON object per line"
    )
    parser.add_argument(
        "--query", "-q", type=str, help="Answer one query in a new conversation and exit"
    )
    parser.add_argument(
        "--user", default=default_user(),
        help="Whose conversations to use (default: $ATLAS_USER or your login name)",
    )
    parser.add_argument(
        "--no-acquire", action="store_true",
        help="Never fetch from trusted sources when the knowledge base falls short",
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("start", help="Start Atlas and query it interactively (the default)")
    commands.add_parser(
        "serve", help="Serve the pipeline over gRPC to the Java API (ATLAS_GRPC_* settings)"
    )
    status = commands.add_parser(
        "status", help="Report the running Atlas and every component, then exit (loads no model)"
    )
    detail = status.add_mutually_exclusive_group()
    detail.add_argument("--time", dest="detail", action="store_const", const="time",
                        help="Stage timings of the service's last query")
    detail.add_argument("--retries", dest="detail", action="store_const", const="retries",
                        help="Planner/critic retries of the service's last query")
    commands.add_parser(
        "conversations", help="List --user's conversations from PostgreSQL, then exit (loads no model)"
    )
    args = parser.parse_args(argv)
    if args.query and args.command not in (None, "start"):
        parser.error(f"--query cannot be combined with {args.command}")
    return args


MONITORS = ("status", "conversations")


def single_query(user_id: str, query: str, acquisition: bool, lease=None) -> int:
    from pipeline.bootstrap import Components

    components = Components(acquisition=acquisition)
    try:
        if lease is not None:
            lease.attach(components)
        pipeline = components.pipeline()
        conversation = pipeline.new_conversation(user_id, query)
        response = pipeline.ask(conversation.conversation_id, user_id, query)
        if response.text is None:
            print(f"Atlas could not answer ({response.record.error_category}): {response.error}",
                  file=sys.stderr)
            return 1
        print(response.text)
        for warning in response.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        return 0
    finally:
        components.close()


def run_service(args) -> int:
    """`start`, `serve` or `-q`: the only paths that load the model, one at a time per data directory."""
    from observability.service_state import AlreadyRunning, ServiceLease

    logger.info("Initializing %s", Config.APP_NAME)
    settings = None
    if args.command == "serve" and not args.query:
        from grpc_server.settings import GrpcSettings, SettingsError

        try:
            settings = GrpcSettings.from_environment()
        except SettingsError as error:
            print(f"Cannot start the gRPC server: {error}", file=sys.stderr)
            return 2
    mode = "query" if args.query else (args.command or "start")
    try:
        lease = ServiceLease(mode).acquire()
    except AlreadyRunning as error:
        print(f"{error}. `main.py status` reports on it.", file=sys.stderr)
        return 1
    try:
        if args.query:
            return single_query(args.user, args.query, not args.no_acquire, lease=lease)
        if args.command == "serve":
            from grpc_server.server import serve

            logger.info("Starting the gRPC server")
            return serve(settings, acquisition=not args.no_acquire, lease=lease)
        from cli.cli_interface import cli_interface

        logger.info("Starting interactive CLI interface for user %s", args.user)
        return cli_interface(args.user, acquisition=not args.no_acquire, lease=lease)
    finally:
        lease.release()


def main(argv=None) -> int:
    """Main runner function."""
    args = parse_arguments(argv)

    configure_logging(
        level=logging.DEBUG if (args.verbose or Config.DEBUG) else logging.INFO,
        console_level=logging.DEBUG if args.verbose else None,
        log_file=args.log_file or Config.LOG_FILE,
        json_lines=args.log_json or None,
    )

    try:
        if args.command in MONITORS:
            from cli import monitor

            if args.command == "status":
                return monitor.status(args.detail)
            return monitor.conversations(args.user)
        return run_service(args)
    except KeyboardInterrupt:
        print("\nGoodbye.")
        return 130
    except Exception as e:
        logger.critical("Unhandled exception in main execution: %s", e, exc_info=True)
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
