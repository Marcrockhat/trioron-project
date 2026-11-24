"""Vee's environment sense: the digital world she lives in, as it is right now
(Rocky s066: "her environment is her digital world … RAM, CPU, clock, IP,
something that only you can see in the WSL").

``read()`` returns one fixed-length vector of readings taken from the machine
at the moment of the call, each mapped to roughly [-1, 1] so no reading
dominates by scale.  Nothing is stored and nothing leaves the machine.  The
host name and the network address are never fed as text or digits: each is
hashed into four numbers, so she can tell "the same place" from "another
place" without the address itself being recoverable from her memory or logs.

Groups (see NAMES):
  clock    time of day and day of week (as points on a circle), hours awake
  memory   RAM used, RAM available, swap used
  cpu      load over 1 and 5 minutes per core, share of time busy right now
  crowd    processes running, processes alive
  disk     share of the disk that is free
  network  links up, a route to the outside, traffic in / out right now
  place    which machine (hashed host name), which address (hashed)
  self     her own process's memory; whether the reflex is loaded

Run from repo root:  python3 experiments/incubator/environment.py     (prints one reading)
"""
from __future__ import annotations
import hashlib, math, os, socket, time

_prev: dict = {}


def _hash4(text: str) -> list[float]:
    d = hashlib.sha256(text.encode()).digest()
    return [d[i] / 127.5 - 1 for i in range(4)]


def _meminfo() -> dict[str, float]:
    return {l.split(":")[0]: float(l.split()[1]) for l in open("/proc/meminfo")}


def _cpu_busy() -> float:
    v = [float(x) for x in open("/proc/stat").readline().split()[1:8]]
    idle, total = v[3] + v[4], sum(v)
    p = _prev.get("cpu")
    _prev["cpu"] = (idle, total)
    return 0.0 if p is None or total == p[1] else 1 - (idle - p[0]) / (total - p[1])


def _net() -> tuple[int, float, float]:
    up = rx = tx = 0
    for l in open("/proc/net/dev").readlines()[2:]:
        name, rest = l.split(":")
        if name.strip() == "lo": continue
        f = rest.split()
        rx += float(f[0]); tx += float(f[8])
        try: up += open(f"/sys/class/net/{name.strip()}/operstate").read().strip() == "up"
        except OSError: pass
    now, p = time.time(), _prev.get("net")
    _prev["net"] = (now, rx, tx)
    if p is None or now <= p[0]: return up, 0.0, 0.0
    return up, (rx - p[1]) / (now - p[0]), (tx - p[2]) / (now - p[0])


def _address() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))          # no packet is sent; this only picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def _route() -> bool:
    try: return any(l.split()[1] == "00000000" for l in open("/proc/net/route").readlines()[1:])
    except OSError: return False


def read(reflex_loaded: bool = False) -> list[float]:
    t = time.localtime()
    day = (t.tm_hour + t.tm_min / 60) / 24
    m = _meminfo()
    load1, load5, _ = os.getloadavg()
    n = os.cpu_count() or 1
    procs = open("/proc/loadavg").read().split()[3].split("/")
    st = os.statvfs("/")
    up, rx, tx = _net()
    rss = float(next(l.split()[1] for l in open("/proc/self/status") if l.startswith("VmRSS")))
    sq = lambda x, top: 2 * min(max(x / top, 0.0), 1.0) - 1          # [0, top] -> [-1, 1]
    lg = lambda x, top: sq(math.log1p(max(x, 0.0)), math.log1p(top))
    return [
        math.sin(2 * math.pi * day), math.cos(2 * math.pi * day),
        math.sin(2 * math.pi * t.tm_wday / 7), math.cos(2 * math.pi * t.tm_wday / 7),
        lg(float(open("/proc/uptime").read().split()[0]) / 3600, 24 * 30),
        sq(1 - m["MemAvailable"] / m["MemTotal"], 1), sq(m["MemAvailable"] / m["MemTotal"], 1),
        sq(1 - m["SwapFree"] / max(m["SwapTotal"], 1), 1),
        sq(load1 / n, 2), sq(load5 / n, 2), sq(_cpu_busy(), 1),
        sq(float(procs[0]), 2 * n), lg(float(procs[1]), 2000),
        sq(st.f_bavail / st.f_blocks, 1),
        sq(up, 4), 1.0 if _route() else -1.0, lg(rx, 1e7), lg(tx, 1e7),
        *_hash4(socket.gethostname()), *_hash4(_address()),
        lg(rss / 1024, 8192), 1.0 if reflex_loaded else -1.0,
    ]


NAMES = ["clock: time of day (sin)", "clock: time of day (cos)", "clock: day of week (sin)", "clock: day of week (cos)",
         "clock: hours the machine has been awake", "memory: RAM used", "memory: RAM available", "memory: swap used",
         "cpu: load 1 min per core", "cpu: load 5 min per core", "cpu: busy right now",
         "crowd: processes running", "crowd: processes alive", "disk: free",
         "network: links up", "network: route to the outside", "network: traffic in", "network: traffic out",
         "place: machine 1", "place: machine 2", "place: machine 3", "place: machine 4",
         "place: address 1", "place: address 2", "place: address 3", "place: address 4",
         "self: my own memory", "self: reflex loaded"]

if __name__ == "__main__":
    read(); time.sleep(1.0)                                            # the "right now" rates need two looks
    for n, v in zip(NAMES, read()):
        print(f"  {n:42} {v:+.2f}")
    print(f"  ({len(NAMES)} readings)")
