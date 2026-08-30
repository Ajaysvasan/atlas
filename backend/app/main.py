import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.cli_interface import cli_interface
from config import Config, get_logger

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
    return parser.parse_args()


def main():
    """
    Main runner function.
    """
    args = parse_arguments()

    # Adjust log level if verbose is set
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    logger.info(f"Initializing {Config.APP_NAME}...")

    try:
        if args.query:
            # Single query execution mode
            logger.info(f"Executing single query: '{args.query}'")
            # For demonstration, we print it directly. You can route this
            # to your query processing logic.
            print(f"Result for '{args.query}': [Processing placeholder]")
        else:
            # Interactive CLI mode
            logger.info("Starting interactive CLI interface...")
            cli_interface()

    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
