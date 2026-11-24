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
  outside  a window to the world beyond the machine — daylight, temperature,
           rain, whether someone is here.  She has no sensor for these: the
           window is CLOSED (all 0) unless the matron shows her something
           through ``simulate``.

``simulate(**over)`` replaces any raw reading (see ``raw``), so the matron can
let her live through a night, a busy or full machine, another place, or the
outside, without waiting for it to happen.

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


def raw(reflex_loaded: bool = False) -> dict:
    """The live readings in plain units (what ``simulate`` can override)."""
    t = time.localtime()
    m = _meminfo()
    load1, load5, _ = os.getloadavg()
    n = os.cpu_count() or 1
    procs = open("/proc/loadavg").read().split()[3].split("/")
    st = os.statvfs("/")
    up, rx, tx = _net()
    return {
        "hour": t.tm_hour + t.tm_min / 60, "weekday": t.tm_wday,
        "awake_hours": float(open("/proc/uptime").read().split()[0]) / 3600,
        "ram_used": 1 - m["MemAvailable"] / m["MemTotal"], "swap_used": 1 - m["SwapFree"] / max(m["SwapTotal"], 1),
        "load1": load1 / n, "load5": load5 / n, "busy": _cpu_busy(),
        "running": float(procs[0]) / (2 * n), "alive": float(procs[1]),
        "disk_free": st.f_bavail / st.f_blocks,
        "links": up, "route": _route(), "traffic_in": rx, "traffic_out": tx,
        "machine": socket.gethostname(), "address": _address(),
        "own_mb": float(next(l.split()[1] for l in open("/proc/self/status") if l.startswith("VmRSS"))) / 1024,
        "reflex": reflex_loaded,
        # the window to the outside: None = closed (nobody has shown her anything)
        "daylight": None, "temperature_c": None, "rain": None, "someone_here": None,
    }


def vector(r: dict) -> list[float]:
    sq = lambda x, top: 2 * min(max(x / top, 0.0), 1.0) - 1          # [0, top] -> [-1, 1]
    lg = lambda x, top: sq(math.log1p(max(x, 0.0)), math.log1p(top))
    win = lambda x, f: 0.0 if x is None else f(x)                    # a closed window reads 0
    day = r["hour"] / 24
    return [
        math.sin(2 * math.pi * day), math.cos(2 * math.pi * day),
        math.sin(2 * math.pi * r["weekday"] / 7), math.cos(2 * math.pi * r["weekday"] / 7),
        lg(r["awake_hours"], 24 * 30),
        sq(r["ram_used"], 1), sq(1 - r["ram_used"], 1), sq(r["swap_used"], 1),
        sq(r["load1"], 2), sq(r["load5"], 2), sq(r["busy"], 1),
        sq(r["running"], 1), lg(r["alive"], 2000),
        sq(r["disk_free"], 1),
        sq(r["links"], 4), 1.0 if r["route"] else -1.0, lg(r["traffic_in"], 1e7), lg(r["traffic_out"], 1e7),
        *_hash4(r["machine"]), *_hash4(r["address"]),
        lg(r["own_mb"], 8192), 1.0 if r["reflex"] else -1.0,
        win(r["daylight"], lambda x: sq(x, 1)), win(r["temperature_c"], lambda x: sq(x + 10, 50)),
        win(r["rain"], lambda x: 1.0 if x else -1.0), win(r["someone_here"], lambda x: 1.0 if x else -1.0),
        0.0 if r["daylight"] is None and r["temperature_c"] is None and r["rain"] is None and r["someone_here"] is None else 1.0,
    ]


def read(reflex_loaded: bool = False) -> list[float]:
    """The machine as it is right now; the window to the outside is closed."""
    return vector(raw(reflex_loaded))


def simulate(**over) -> list[float]:
    """The live reading with any raw value replaced, e.g.
    ``simulate(hour=2, ram_used=.9, machine="another-pc", daylight=0, rain=True)``:
    the matron shows her a night, a full memory, another place, or the outside."""
    r = raw()
    unknown = set(over) - set(r)
    if unknown: raise KeyError(f"not a reading: {sorted(unknown)}")
    return vector({**r, **over})


NAMES = ["clock: time of day (sin)", "clock: time of day (cos)", "clock: day of week (sin)", "clock: day of week (cos)",
         "clock: hours the machine has been awake", "memory: RAM used", "memory: RAM available", "memory: swap used",
         "cpu: load 1 min per core", "cpu: load 5 min per core", "cpu: busy right now",
         "crowd: processes running", "crowd: processes alive", "disk: free",
         "network: links up", "network: route to the outside", "network: traffic in", "network: traffic out",
         "place: machine 1", "place: machine 2", "place: machine 3", "place: machine 4",
         "place: address 1", "place: address 2", "place: address 3", "place: address 4",
         "self: my own memory", "self: reflex loaded",
         "outside: daylight", "outside: temperature", "outside: rain", "outside: someone is here", "outside: window open"]

if __name__ == "__main__":
    read(); time.sleep(1.0)                                            # the "right now" rates need two looks
    night = simulate(hour=2, ram_used=0.9, busy=0.95, machine="another-pc", daylight=0.0, temperature_c=18, rain=True, someone_here=False)
    live = read()
    assert len(live) == len(night) == len(NAMES)
    print(f"  {'reading':42} {'now':>6} {'simulated rainy night, busy, elsewhere':>40}")
    for n, v, w in zip(NAMES, live, night):
        print(f"  {n:42} {v:+6.2f} {w:+40.2f}")
    print(f"  ({len(NAMES)} readings)")
