from enum import Enum, auto
from traceback import format_exc
from typing import Callable, Optional


class Signal(Enum):
    NONE = auto()
    SHOW_ERROR_MESSAGE = auto()


_UUID_COUNTER: int = 0
_REGISTERED_CALLBACKS: dict[Signal, list[tuple[Callable, int]]] = {}
_QUEUE: Optional[dict[Signal, list[tuple[tuple, dict]]]] = {}


def emit(signal: Signal, *args, **kwargs):
    global _REGISTERED_CALLBACKS
    global _QUEUE

    assert type(signal) is Signal, signal

    if _QUEUE is not None and signal not in _REGISTERED_CALLBACKS:
        if signal not in _QUEUE:
            _QUEUE[signal] = []
        _QUEUE[signal].append(
            (
                args,
                kwargs,
            )
        )
        return

    assert signal in _REGISTERED_CALLBACKS, signal

    try:
        func: Callable
        uuid: int
        for (func, uuid) in _REGISTERED_CALLBACKS[signal]:
            func(*args, **kwargs)
    except Exception:
        if signal == Signal.SHOW_ERROR_MESSAGE:
            print(format_exc())
        else:
            emit(Signal.SHOW_ERROR_MESSAGE, format_exc())


def register(signal: Signal, callback: Callable) -> int:
    global _UUID_COUNTER
    global _REGISTERED_CALLBACKS

    assert type(signal) is Signal, signal

    try:
        if signal not in _REGISTERED_CALLBACKS:
            _REGISTERED_CALLBACKS[signal] = []

        _UUID_COUNTER += 1
        _REGISTERED_CALLBACKS[signal].append(
            (
                callback,
                _UUID_COUNTER,
            )
        )
        return _UUID_COUNTER

    except Exception:
        if signal == Signal.SHOW_ERROR_MESSAGE:
            print(format_exc())
        else:
            emit(Signal.SHOW_ERROR_MESSAGE, format_exc())

        return -1


def unregister(
        signal: Signal,
        callback: Optional[Callable] = None,
        uuid: Optional[int] = None,
):
    global _REGISTERED_CALLBACKS

    assert type(signal) is Signal, signal
    assert callback is not None or type(uuid) is int and uuid > 0, (
        callback,
        uuid,
    )
    assert signal in _REGISTERED_CALLBACKS, signal

    try:
        chosen_entry: Optional[tuple[Callable, int]] = None
        entry: tuple[Callable, int]
        for entry in _REGISTERED_CALLBACKS[signal]:
            if entry[0] == callback or entry[1] == uuid:
                chosen_entry = entry
                break

        assert chosen_entry is not None

        _REGISTERED_CALLBACKS[signal].remove(chosen_entry)
    except Exception:
        if signal == Signal.SHOW_ERROR_MESSAGE:
            print(format_exc())
        else:
            emit(Signal.SHOW_ERROR_MESSAGE, format_exc())


def clear(signal: Signal):
    global _REGISTERED_CALLBACKS

    assert type(signal) is Signal, signal
    assert signal in _REGISTERED_CALLBACKS, signal

    try:
        del _REGISTERED_CALLBACKS[signal]
    except Exception:
        if signal == Signal.SHOW_ERROR_MESSAGE:
            print(format_exc())
        else:
            emit(Signal.SHOW_ERROR_MESSAGE, format_exc())


def emit_backlog():
    global _QUEUE

    signal: Signal
    queue: list[tuple[list, dict]]
    for signal, queue in _QUEUE.items():
        for (args, kwargs) in queue:
            emit(signal, *args, **kwargs)

    _QUEUE = None
