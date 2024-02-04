import heapq


class TTLCounter:
    """TTLCounter is a counter/list that expires items after a TTL period expires."""

    def __init__(self, max_ttl: int) -> None:
        self.inner_counter = []
        self.max_ttl = max_ttl

    def __len__(self):
        """Return the number of items in the counter."""
        return len(self.inner_counter)

    def add_item(self, hash: str):
        """Add an item with the max TTL."""
        heapq.heappush(self.inner_counter, (self.max_ttl, hash))

    def tick(self):
        """Decrease the TTL of all items by 1."""
        for i, (ttl, hash) in enumerate(self.inner_counter):
            self.inner_counter[i] = (ttl - 1, hash)

    def expire_one(self):
        """Expire the first item if its TTL is 0. Expires AT MOST one item."""
        # peek the first item
        ttl, hash = self.inner_counter[0]
        if ttl <= 0:
            heapq.heappop(self.inner_counter)
            return hash
        return None

    def expire_all(self):
        """Expire all items."""
        for _, hash in self.inner_counter:
            yield hash
        self.inner_counter.clear()
