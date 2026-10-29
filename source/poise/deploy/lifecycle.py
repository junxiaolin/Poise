"""Stop hand control before waiting on capture or camera shutdown."""

from contextlib import ExitStack, contextmanager
import logging


@contextmanager
def deployment_resources(client):
    """Register camera then capture contexts; robot shutdown always runs first.

    Even if ESTOP communication fails, closing the helper triggers its independent
    watchdog. Cleanup failures are logged without replacing the original fault.
    """
    resources = ExitStack()
    fault = None
    try:
        yield resources
    except BaseException as error:
        fault = error
        if client.started and client.status.armed:
            try:
                client.estop("host_control_failure")
            except BaseException:
                logging.exception("Failed to acknowledge robot stop during shutdown")
        raise
    finally:
        try:
            client.close()
        except BaseException as error:
            if fault is None:
                fault = error
                raise
            logging.exception("Failed to close SDK helper after control failure")
        finally:
            try:
                resources.close()
            except BaseException:
                if fault is None:
                    raise
                logging.exception("Failed to close capture/camera after control failure")
