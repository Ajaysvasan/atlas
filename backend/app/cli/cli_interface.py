from config import get_logger, log_context, new_correlation_id

logger = get_logger(__name__)


def cli_interface():
    logger.info("CLI session started.")
    try:
        while True:
            query = input("Enter the query (Press ctrl + c to exit) : ")
            if query.lower() == "exit":
                logger.info("User requested exit from CLI session.")
                print("Exiting the system. Goodbye!")
                break
            # One id per query, bound for the whole turn. Every module that
            # logs while answering it carries the id, which is what lets a
            # single request be pulled out of a log the planner's concurrent
            # nodes are writing to at the same time.
            with log_context(query_id=new_correlation_id()):
                logger.info("Received user query: %r", query)
                print(f"Processing query: {query}")
                logger.debug("Executing query pipeline")
                # some stuff
    except KeyboardInterrupt:
        logger.info("CLI session interrupted by user (KeyboardInterrupt).")
        print("\nExiting the system. Goodbye!")
    except Exception as e:
        logger.error("Error encountered during CLI session: %s", e, exc_info=True)
        raise
