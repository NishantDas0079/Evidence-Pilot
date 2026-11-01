import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Per-IP hourly and global daily caps. In-memory: resets on restart, which is fine here."""

    def __init__(self, per_ip_per_hour: int, global_per_day: int, clock=time.time):
        self.per_ip = per_ip_per_hour
        self.global_cap = global_per_day
        self._clock = clock
        self._ip: dict[str, deque] = defaultdict(deque)
        self._all: deque = deque()
        self._lock = threading.Lock()

    def check(self, ip: str) -> str | None:
        """Record the request and return None, or return the reason it is refused."""
        now = self._clock()
        with self._lock:
            while self._all and now - self._all[0] > 86400:
                self._all.popleft()
            hits = self._ip[ip]
            while hits and now - hits[0] > 3600:
                hits.popleft()
            if len(self._all) >= self.global_cap:
                return "The demo has reached its daily audit limit. Please try again tomorrow."
            if len(hits) >= self.per_ip:
                return "Too many audits from your address this hour. Please wait a while and try again."
            hits.append(now)
            self._all.append(now)
            return None
