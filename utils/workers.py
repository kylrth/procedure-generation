import asyncio
from collections.abc import Iterable
from typing import Awaitable, Callable, TypeVar

from tqdm import tqdm


class _Canceled:
    """Shows up in the queue to tell a worker to shut down."""


T = TypeVar("T")
U = TypeVar("U")


async def spread_gather(
    func: Callable[[T], Awaitable[U]], data: Iterable[T], n: int, length: int | None = None
) -> list[U]:
    """Apply func to all items in data using n concurrent workers, and then return the list of
    results. The results won't necessarily be in order.

    If length is provided, it is used to provide a tqdm progress bar.
    """
    queue = asyncio.Queue()
    results = []

    progress = tqdm(total=length) if length else None

    # define a worker that pulls from the queue until it receives a _Canceled
    async def _worker():
        while True:
            item = await queue.get()
            try:
                if isinstance(item, _Canceled):
                    return
                result = await func(item)
                results.append(result)
                if progress:
                    progress.update(1)
            finally:
                queue.task_done()

    # start workers
    w = []
    for _ in range(n):
        w.append(asyncio.create_task(_worker()))

    # enqueue items
    for item in data:
        await queue.put(item)

    # signal workers to exit at the end of the queue
    for _ in range(n):
        await queue.put(_Canceled())

    # surface any exceptions from workers
    await asyncio.gather(*w)

    # make sure the queue is empty (there's no way it isn't, but we'll just check)
    await queue.join()

    if progress:
        progress.close()

    return results
