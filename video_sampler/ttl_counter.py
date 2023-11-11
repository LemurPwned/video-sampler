import heapq


class TTLCounter:
    def __init__(self, max_ttl: int) -> None:
        self.inner_counter = []
        self.max_ttl = max_ttl

    def __len__(self):
        return len(self.inner_counter)

    def add_item(self, hash: str):
        heapq.heappush(self.inner_counter, (self.max_ttl, hash))

    def tick(self):
        for i, (ttl, hash) in enumerate(self.inner_counter):
            self.inner_counter[i] = (ttl - 1, hash)

    def expire_one(self):
        # peek the first item
        ttl, hash = self.inner_counter[0]
        if ttl <= 0:
            heapq.heappop(self.inner_counter)
            return hash
        return None

    def expire_all(self):
        for _, hash in self.inner_counter:
            yield hash
        self.inner_counter.clear()
