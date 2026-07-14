from threading import Event


class GenerationCancelledError(RuntimeError):
    """Raised when a user requests cancellation of an active generation job."""


def raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("Generation job was cancelled by the user")
