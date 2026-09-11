import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.cli_interface import cli_interface
from config import Config, configure_logging, get_logger, log_context, new_correlation_id

logger = get_logger("backend_main")


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Final Year Project Backend - Command Line Interface"
    )
    parser.add_argument(
        "--query", "-q", type=str, help="Run a single query directly and exit"
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
    return parser.parse_args()


def main():
    """
    Main runner function.
    """
    args = parse_arguments()

    # The one place handlers are installed, and the reason --verbose now reaches
    # the whole application: the level is set on the root logger before any
    # module logs, and every module logger propagates to it. Previously the flag
    # was applied to this file's logger alone, so data_layer and memory stayed
    # at INFO no matter what was passed.
    configure_logging(
        level=logging.DEBUG if (args.verbose or Config.DEBUG) else logging.INFO,
        console_level=logging.DEBUG if args.verbose else None,
        log_file=args.log_file or Config.LOG_FILE,
        json_lines=args.log_json or None,
    )

    logger.info("Initializing %s", Config.APP_NAME)

    try:
        if args.query:
            # Single query execution mode
            with log_context(query_id=new_correlation_id()):
                logger.info("Executing single query: %r", args.query)
                # For demonstration, we print it directly. You can route this
                # to your query processing logic.
                print(f"Result for '{args.query}': [Processing placeholder]")
        else:
            # Interactive CLI mode
            logger.info("Starting interactive CLI interface")
            cli_interface()

    except Exception as e:
        logger.critical("Unhandled exception in main execution: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
