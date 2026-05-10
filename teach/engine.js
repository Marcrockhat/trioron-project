// trioron teaching petri dish — Canvas engine
// Each Trioron is a visible cell living in (valence, arousal) space.
// Canvas x = valence, y = arousal (canvas y is flipped, +arousal up).

const POPULATION_CAP_DEFAULT = 200;
const FISSION_COOLDOWN_TICKS = 90;
const MIN_LIVE_POP = 5;            // dream / cap-turnover never reduce alive non-donor cells below this

const T1_LABELS = {
  HAPPY: { hue: 50, vx: 0.7, vy: 0.0 },
  SAD:   { hue: 230, vx: -0.7, vy: 0.0 },
};
const T2_LABELS = {
  ANGRY: { hue: 0,   vx: 0.0, vy: 0.7 },
  CALM:  { hue: 130, vx: 0.0, vy: -0.7 },
};

const LINEAGE_HOME = 200;       // genesis lineage hue (cool blue inner ring)
const LINEAGE_FOREIGN = 35;     // foreign donor lineage hue (warm orange)

class Trioron {
  // substrateWidth = L0 latent dim (shared substrate; sets inner ring radius)
  // branchWidth    = L1 branch this cell carries on top of L0 (0 for genesis)
  // outer body radius = inner ring radius + branchWidth contribution
  constructor({ x, y, seedColor, substrateWidth, branchWidth = 0, specialty = null, isDonor = false, donorTarget = null }) {
    this.x = x;
    this.y = y;
    this.vx = 0;
    this.vy = 0;
    this.seedColor = seedColor;
    this.substrateWidth = substrateWidth;
    this.branchWidth = branchWidth;
    this.specialty = specialty;
    this.frustration = 0;
    this.firingRecency = 0;
    this.age = 0;
    this.alive = true;
    this.fissionCooldown = 0;
    this.spawnAnim = 1.0;
    this.isDonor = isDonor;
    this.donorTarget = donorTarget;
    this.docked = false;
    this.dockProgress = 0;
    this.fading = false;
    this.fadeProgress = 0;
    this.corruption = 0;
  }

  innerRingRadius() {
    return Math.max(2.5, 2 + this.substrateWidth * 0.5);
  }

  bodyRadius() {
    return this.innerRingRadius() + this.branchWidth * 0.5;
  }

  // back-compat alias used by repulsion physics
  outerRadius() {
    return this.bodyRadius();
  }

  effectiveSpecialty(world) {
    if (this.specialty != null) return this.specialty;
    return positionToHue(world, this.x, this.y);
  }

  draw(ctx, world) {
    let alpha = 1.0;
    if (this.fading) alpha = Math.max(0, 1 - this.fadeProgress);
    if (this.spawnAnim > 0) alpha = Math.min(alpha, 1 - 0.5 * this.spawnAnim);
    if (this.docked) alpha = Math.max(0, 1 - this.dockProgress);

    const scale = Math.max(0.1, 1.0 - 0.5 * this.spawnAnim - 0.5 * this.dockProgress);
    const ir = this.innerRingRadius() * scale;
    const r = this.bodyRadius() * scale;
    if (r < 0.5) return;
    const hue = this.effectiveSpecialty(world);
    const sat = Math.max(8, 32 - this.corruption * 24);

    if (this.frustration > 0.05 && !this.fading) {
      const glowR = r + 4 + this.frustration * 14;
      const grad = ctx.createRadialGradient(this.x, this.y, r, this.x, this.y, glowR);
      grad.addColorStop(0, `rgba(230, 80, 80, ${(0.35 * this.frustration * alpha).toFixed(3)})`);
      grad.addColorStop(1, "rgba(230, 80, 80, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(this.x, this.y, glowR, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = alpha;

    // body
    ctx.beginPath();
    ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${hue}, ${sat}%, 58%)`;
    ctx.fill();
    ctx.strokeStyle = `hsla(${hue}, 50%, 28%, 0.9)`;
    ctx.lineWidth = 1;
    ctx.stroke();

    // inner ring (lineage)
    ctx.beginPath();
    ctx.arc(this.x, this.y, ir, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${this.seedColor}, 75%, 55%)`;
    ctx.fill();

    // donor: small halo + a larger dashed absorption_radius circle that the slider controls
    if (this.isDonor && !this.docked && !this.fading) {
      ctx.beginPath();
      ctx.arc(this.x, this.y, r + 4, 0, Math.PI * 2);
      ctx.strokeStyle = `hsla(${this.seedColor}, 60%, 70%, 0.7)`;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(this.x, this.y, world.params.absorption_radius, 0, Math.PI * 2);
      ctx.strokeStyle = `hsla(${this.seedColor}, 50%, 65%, 0.28)`;
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 8]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.globalAlpha = 1.0;
  }

  tick(world) {
    if (this.fading) {
      this.fadeProgress = Math.min(1, this.fadeProgress + 0.025);
      if (this.fadeProgress >= 1) this.alive = false;
      return;
    }
    if (this.docked) {
      this.dockProgress = Math.min(1, this.dockProgress + 0.04);
      if (this.dockProgress >= 1) this.alive = false;
      return;
    }

    if (this.isDonor) {
      // donor drifts toward target
      if (this.donorTarget) {
        const dx = this.donorTarget.x - this.x;
        const dy = this.donorTarget.y - this.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d > 1) {
          this.vx += (dx / d) * 0.06;
          this.vy += (dy / d) * 0.06;
        }
      }
    } else if (!world.dreaming) {
      // Brownian wander when not dreaming
      this.vx += (Math.random() - 0.5) * 0.04;
      this.vy += (Math.random() - 0.5) * 0.04;
    }

    // soft repulsion vs neighbors
    for (const o of world.triorons) {
      if (o === this || !o.alive || o.fading) continue;
      const dx = this.x - o.x, dy = this.y - o.y;
      const d2 = dx * dx + dy * dy;
      const minD = (this.outerRadius() + o.outerRadius()) * 1.4;
      if (d2 < minD * minD && d2 > 0.01) {
        const d = Math.sqrt(d2);
        const push = (minD - d) * 0.02;
        this.vx += (dx / d) * push;
        this.vy += (dy / d) * push;
      }
    }

    this.vx *= world.dreaming ? 0.6 : 0.88;
    this.vy *= world.dreaming ? 0.6 : 0.88;
    this.x += this.vx;
    this.y += this.vy;

    const m = 12;
    if (this.x < m) { this.x = m; this.vx *= -0.4; }
    if (this.x > world.width - m) { this.x = world.width - m; this.vx *= -0.4; }
    if (this.y < m) { this.y = m; this.vy *= -0.4; }
    if (this.y > world.height - m) { this.y = world.height - m; this.vy *= -0.4; }

    this.frustration *= 0.992;
    if (this.fissionCooldown > 0) this.fissionCooldown--;
    if (this.spawnAnim > 0) this.spawnAnim = Math.max(0, this.spawnAnim - 0.04);
    this.firingRecency++;
    this.age++;
  }

  canFission(threshold) {
    if (this.isDonor || this.fading || this.docked) return false;
    return this.frustration > threshold && this.fissionCooldown <= 0;
  }
}

class DataPoint {
  constructor({ x, y, label, ghost = false }) {
    this.x = x;
    this.y = y;
    this.label = label;
    this.ghost = ghost;
    this.alive = true;
    this.age = 0;
    this.fadeColor = null;
    this.fadeAge = 0;
  }

  draw(ctx) {
    const baseAlpha = this.ghost ? 0.35 : 1.0;
    if (this.fadeColor) {
      const alpha = Math.max(0, 1 - this.fadeAge / 30) * baseAlpha;
      ctx.beginPath();
      ctx.arc(this.x, this.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = this.fadeColor.replace("ALPHA", alpha.toFixed(2));
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.ghost ? 2 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${this.label}, 80%, 65%, ${baseAlpha})`;
      ctx.fill();
      if (this.ghost) {
        ctx.strokeStyle = `hsla(${this.label}, 40%, 60%, 0.5)`;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  }

  tick() {
    this.age++;
    if (this.fadeColor) {
      this.fadeAge++;
      if (this.fadeAge > 30) this.alive = false;
    } else if (this.age > 240) {
      this.alive = false;
    }
  }
}

function positionToHue(world, x, y) {
  const cx = world.width / 2, cy = world.height / 2;
  const valence = (x - cx) / (world.width / 2);
  const arousal = (cy - y) / (world.height / 2);
  const theta = Math.atan2(arousal, valence);
  const deg = (theta * 180 / Math.PI + 360) % 360;
  return (50 - deg + 360) % 360;
}

function hueDist(a, b) {
  let d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

function spawnDataPoint(world, label, ghost = false) {
  const cx = world.width / 2, cy = world.height / 2;
  const jx = (Math.random() + Math.random() + Math.random() - 1.5) * 60;
  const jy = (Math.random() + Math.random() + Math.random() - 1.5) * 60;
  const x = cx + label.vx * (world.width / 2) + jx;
  const y = cy - label.vy * (world.height / 2) + jy;
  world.dataPoints.push(new DataPoint({ x, y, label: label.hue, ghost }));
}

function spawnT1Point(world, ghost = false) {
  const k = Math.random() < 0.5 ? "HAPPY" : "SAD";
  spawnDataPoint(world, T1_LABELS[k], ghost);
}

function spawnT2Point(world, ghost = false) {
  const k = Math.random() < 0.5 ? "ANGRY" : "CALM";
  spawnDataPoint(world, T2_LABELS[k], ghost);
}

function runStream(world, ratePerTick, t2 = false, replayLambda = 0) {
  if (Math.random() < ratePerTick) {
    if (t2) spawnT2Point(world, false);
    else spawnT1Point(world, false);
  }
  // replay: probability λ each tick, spawn a ghost point of the *other* task
  if (replayLambda > 0 && Math.random() < replayLambda) {
    if (t2) spawnT1Point(world, true);   // T2 active → ghost T1
    else spawnT2Point(world, true);      // (symmetric, unused atm)
  }
  for (const p of world.dataPoints) {
    if (!p.fadeColor && p.age > 8) world.classifyPoint(p);
  }
}

class World {
  constructor(dishCanvas, insetCanvas) {
    this.canvas = dishCanvas;
    this.ctx = dishCanvas.getContext("2d");
    this.width = dishCanvas.width;
    this.height = dishCanvas.height;
    this.inset = insetCanvas;
    this.insetCtx = insetCanvas.getContext("2d");
    this.iw = insetCanvas.width;
    this.ih = insetCanvas.height;

    this.triorons = [];
    this.dataPoints = [];
    this.flashes = [];
    this.dockLines = [];        // visual links during docking / dream

    this.params = {
      l0_dim: 8,
      frustration_threshold: 0.4,
      branch_width: 8,
      replay_lambda: 0.4,
      max_downscales_per_layer: 8,
      absorption_radius: 60,
      seed_match_required: 1,         // 0 = false, 1 = true
      population_cap: POPULATION_CAP_DEFAULT,
      manifold_noise_scale: 1.0,
    };

    this.scene = null;
    this.tickN = 0;
    this.dreaming = false;
    this.dreamTicks = 0;
    this.dreamFlares = 0;

    // simulation speed — 1.0 = native, <1 = slow-motion, >1 = fast-forward
    this.speed = 1.0;
    this._tickAccum = 0;

    this.onClick = null;
    dishCanvas.addEventListener("click", (e) => {
      const rect = dishCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (this.width / rect.width);
      const y = (e.clientY - rect.top) * (this.height / rect.height);
      if (this.onClick) this.onClick(x, y);
    });
  }

  reset() {
    this.triorons = [];
    this.dataPoints = [];
    this.flashes = [];
    this.dockLines = [];
    this.tickN = 0;
    this.dreaming = false;
    this.dreamTicks = 0;
    this.dreamFlares = 0;
    this.onClick = null;
  }

  // Deep-copy world state. Includes a snapshot of params so each scene
  // remembers its own knob value across navigation.
  snapshot() {
    const cloneT = t => Object.assign(Object.create(Trioron.prototype), t);
    const cloneP = p => Object.assign(Object.create(DataPoint.prototype), p);
    return {
      triorons: this.triorons.map(cloneT),
      dataPoints: this.dataPoints.map(cloneP),
      flashes: this.flashes.map(f => ({ ...f })),
      dockLines: this.dockLines.map(l => ({ ...l })),
      dreaming: this.dreaming,
      dreamTicks: this.dreamTicks,
      dreamFlares: this.dreamFlares,
      tickN: this.tickN,
      params: { ...this.params },
    };
  }

  restore(snap) {
    const cloneT = t => Object.assign(Object.create(Trioron.prototype), t);
    const cloneP = p => Object.assign(Object.create(DataPoint.prototype), p);
    this.triorons = snap.triorons.map(cloneT);
    this.dataPoints = snap.dataPoints.map(cloneP);
    this.flashes = snap.flashes.map(f => ({ ...f }));
    this.dockLines = snap.dockLines.map(l => ({ ...l }));
    this.dreaming = snap.dreaming;
    this.dreamTicks = snap.dreamTicks;
    this.dreamFlares = snap.dreamFlares;
    this.tickN = snap.tickN;
    if (snap.params) this.params = { ...snap.params };
    this.onClick = null;
  }

  // Run physics + classification + fission off-screen until target population is reached.
  // Used to ensure scenes that depend on a healthy population have one at entry.
  warmupTo(targetPop, taskFn, maxTicks = 600) {
    let n = 0;
    const livePop = () => this.triorons.filter(t => t.alive && !t.fading && !t.isDonor).length;
    while (livePop() < targetPop && n < maxTicks) {
      taskFn(this);
      for (const t of this.triorons) t.tick(this);
      for (const p of this.dataPoints) p.tick();
      this.dataPoints = this.dataPoints.filter(p => p.alive);
      const thr = this.params.frustration_threshold;
      const toFission = this.triorons.filter(t => t.canFission(thr));
      for (const parent of toFission) this.fission(parent);
      this.triorons = this.triorons.filter(t => t.alive);
      n++;
    }
  }

  spawnGenesis() {
    // genesis is pure substrate — L0 only, no branch yet.
    const t = new Trioron({
      x: this.width / 2,
      y: this.height / 2,
      seedColor: LINEAGE_HOME,
      substrateWidth: this.params.l0_dim,
      branchWidth: 0,
    });
    t.spawnAnim = 1.0;
    this.triorons.push(t);
  }

  classifyPoint(point) {
    const live = this.triorons.filter(t => t.alive && !t.fading && !t.docked && !t.isDonor);
    if (live.length === 0) return;
    let nearest = null, ndist = Infinity;
    for (const t of live) {
      const dx = t.x - point.x, dy = t.y - point.y;
      const d = dx * dx + dy * dy;
      if (d < ndist) { ndist = d; nearest = t; }
    }
    if (!nearest) return;

    nearest.firingRecency = 0;
    const spec = nearest.effectiveSpecialty(this);
    const hd = hueDist(spec, point.label);
    const correct = hd < 50;

    if (correct) {
      point.fadeColor = `hsla(${point.label}, 70%, 60%, ALPHA)`;
      // gentle drift toward labeled point (tiny "online learning")
      const pull = point.ghost ? 0.02 : 0.05;
      nearest.vx += (point.x - nearest.x) * pull * 0.05;
      nearest.vy += (point.y - nearest.y) * pull * 0.05;
    } else {
      point.fadeColor = `hsla(0, 0%, 50%, ALPHA)`;
      const frustGain = point.ghost ? 0.025 : 0.06;
      nearest.frustration = Math.min(1.0, nearest.frustration + frustGain);
    }
  }

  fission(parent) {
    const liveCount = this.triorons.filter(t => t.alive && !t.fading && !t.isDonor).length;
    if (liveCount >= this.params.population_cap) {
      // apoptosis-driven turnover: kill the quietest cell first.
      // Floor at MIN_LIVE so we never pinch the population to nothing.
      if (liveCount <= MIN_LIVE_POP) return;
      const quiet = this.triorons
        .filter(t => t.alive && !t.fading && !t.isDonor && t !== parent)
        .sort((a, b) => b.firingRecency - a.firingRecency)[0];
      if (quiet && quiet.firingRecency > 60) {
        quiet.fading = true;
      } else {
        return;
      }
    }
    const angle = Math.random() * Math.PI * 2;
    const dist = parent.outerRadius() * 1.6;
    // child inherits the substrate and grows its own branch.
    // parent also grows a branch of its own (it's now "L0 + branch").
    const child = new Trioron({
      x: parent.x + Math.cos(angle) * dist,
      y: parent.y + Math.sin(angle) * dist,
      seedColor: parent.seedColor,
      substrateWidth: this.params.l0_dim,
      branchWidth: this.params.branch_width,
    });
    if (parent.branchWidth === 0) {
      // first fission of genesis — genesis itself becomes a branched unit
      parent.branchWidth = this.params.branch_width;
    }
    child.vx = Math.cos(angle) * 1.2;
    child.vy = Math.sin(angle) * 1.2;
    parent.vx -= Math.cos(angle) * 0.6;
    parent.vy -= Math.sin(angle) * 0.6;
    parent.frustration = 0;
    parent.fissionCooldown = FISSION_COOLDOWN_TICKS;
    this.triorons.push(child);
    this.flashes.push({ x: parent.x, y: parent.y, age: 0, max: 24, color: "246, 201, 113" });
  }

  spawnDonor({ seedColor, specialty }) {
    // Spawn at random edge with a target near the population centroid
    const side = Math.floor(Math.random() * 4);
    let x, y;
    const m = 20;
    if (side === 0)      { x = m;             y = Math.random() * this.height; }
    else if (side === 1) { x = this.width - m; y = Math.random() * this.height; }
    else if (side === 2) { x = Math.random() * this.width;  y = m; }
    else                 { x = Math.random() * this.width;  y = this.height - m; }

    const live = this.triorons.filter(t => t.alive && !t.fading && !t.isDonor);
    let tx = this.width / 2, ty = this.height / 2;
    if (live.length > 0) {
      tx = live.reduce((s, t) => s + t.x, 0) / live.length;
      ty = live.reduce((s, t) => s + t.y, 0) / live.length;
    }
    const donor = new Trioron({
      x, y,
      seedColor,
      substrateWidth: this.params.l0_dim,
      branchWidth: this.params.branch_width * 1.4,
      specialty,
      isDonor: true,
      donorTarget: { x: tx, y: ty },
    });
    donor.spawnAnim = 1.0;
    this.triorons.push(donor);
    return donor;
  }

  tickDonors() {
    const seedMatch = this.params.seed_match_required >= 0.5;
    for (const d of this.triorons) {
      if (!d.isDonor || d.docked || d.fading || !d.alive) continue;
      // find nearest non-donor host
      let host = null, ndist = Infinity;
      for (const t of this.triorons) {
        if (t === d || t.isDonor || !t.alive || t.fading) continue;
        const dx = t.x - d.x, dy = t.y - d.y;
        const dd = dx * dx + dy * dy;
        if (dd < ndist) { ndist = dd; host = t; }
      }
      if (!host) continue;
      const dist = Math.sqrt(ndist);
      const radius = this.params.absorption_radius;
      if (dist < radius) {
        const compatible = d.seedColor === host.seedColor;
        if (compatible) this.absorbCompatible(host, d);
        else if (seedMatch) this.bounceForeign(host, d);
        else this.absorbForcedCorrupt(host, d);
      }
    }
  }

  absorbCompatible(host, donor) {
    donor.docked = true;
    // host gains half the donor's branch on top of its own
    host.branchWidth += donor.branchWidth * 0.5;
    host.spawnAnim = 0.6;
    this.flashes.push({ x: host.x, y: host.y, age: 0, max: 30, color: "150, 220, 180" });
    this.dockLines.push({ x1: donor.x, y1: donor.y, x2: host.x, y2: host.y, age: 0, max: 30, color: "150, 220, 180" });
  }

  bounceForeign(host, donor) {
    const dx = donor.x - host.x, dy = donor.y - host.y;
    const d = Math.max(0.01, Math.sqrt(dx * dx + dy * dy));
    donor.vx = (dx / d) * 4;
    donor.vy = (dy / d) * 4;
    donor.donorTarget = null;        // give up trying
    donor.frustration = 1.0;
    host.frustration = Math.min(1.0, host.frustration + 0.3);
    this.flashes.push({ x: (host.x + donor.x) / 2, y: (host.y + donor.y) / 2, age: 0, max: 22, color: "230, 80, 80" });
  }

  absorbForcedCorrupt(host, donor) {
    donor.docked = true;
    host.branchWidth += donor.branchWidth * 0.5;
    host.corruption = Math.min(1.0, host.corruption + 0.7);
    host.frustration = Math.min(1.0, host.frustration + 0.5);
    this.flashes.push({ x: host.x, y: host.y, age: 0, max: 30, color: "230, 140, 80" });
    this.dockLines.push({ x1: donor.x, y1: donor.y, x2: host.x, y2: host.y, age: 0, max: 30, color: "230, 140, 80" });
  }

  startDream() {
    this.dreaming = true;
    this.dreamTicks = 0;
    this.dreamFlares = 0;
  }

  tickDream() {
    this.dreamTicks++;
    const cap = this.params.max_downscales_per_layer;
    if (this.dreamTicks % 8 === 0 && this.dreamFlares < cap) {
      // pick a random pair of nearby live cells, flare between them, downscale frustration
      const live = this.triorons.filter(t => t.alive && !t.fading && !t.isDonor && !t.docked);
      if (live.length >= 2) {
        const a = live[Math.floor(Math.random() * live.length)];
        const candidates = live.filter(t => t !== a);
        candidates.sort((p, q) => {
          const dp = (p.x - a.x) ** 2 + (p.y - a.y) ** 2;
          const dq = (q.x - a.x) ** 2 + (q.y - a.y) ** 2;
          return dp - dq;
        });
        const b = candidates[Math.min(2, candidates.length - 1)];   // 3rd nearest, not closest
        if (b) {
          this.dockLines.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, age: 0, max: 24, color: "180, 200, 255" });
          a.frustration *= 0.5;
          b.frustration *= 0.5;
          this.dreamFlares++;
        }
      }
    }
    // apoptosis: cells that haven't fired in a while fade out — but only
    // if we have headroom above MIN_LIVE_POP so dream doesn't depopulate small worlds.
    if (this.dreamTicks === 60) {
      const live = this.triorons.filter(t => t.alive && !t.fading && !t.isDonor);
      const headroom = Math.max(0, live.length - MIN_LIVE_POP);
      if (headroom > 0) {
        const stale = live
          .filter(t => t.firingRecency > 200)
          .sort((a, b) => b.firingRecency - a.firingRecency);
        for (let i = 0; i < Math.min(2, stale.length, headroom); i++) {
          stale[i].fading = true;
        }
      }
    }
    if (this.dreamTicks > 150) this.dreaming = false;
  }

  drawAxes() {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(this.width / 2, 0);
    ctx.lineTo(this.width / 2, this.height);
    ctx.moveTo(0, this.height / 2);
    ctx.lineTo(this.width, this.height / 2);
    ctx.stroke();

    ctx.fillStyle = "rgba(200,200,200,0.35)";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("valence →", this.width - 70, this.height / 2 - 6);
    ctx.fillText("← valence", 8, this.height / 2 - 6);
    ctx.fillText("↑ arousal", this.width / 2 + 8, 14);
    ctx.fillText("↓ arousal", this.width / 2 + 8, this.height - 6);
    ctx.restore();
  }

  drawFlashes() {
    for (const f of this.flashes) {
      const t = f.age / f.max;
      const r = 6 + t * 30;
      const a = 0.5 * (1 - t);
      this.ctx.beginPath();
      this.ctx.arc(f.x, f.y, r, 0, Math.PI * 2);
      this.ctx.strokeStyle = `rgba(${f.color}, ${a.toFixed(2)})`;
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
      f.age++;
    }
    this.flashes = this.flashes.filter(f => f.age < f.max);
  }

  drawDockLines() {
    for (const l of this.dockLines) {
      const t = l.age / l.max;
      const a = (1 - t) * 0.7;
      this.ctx.beginPath();
      this.ctx.moveTo(l.x1, l.y1);
      this.ctx.lineTo(l.x2, l.y2);
      this.ctx.strokeStyle = `rgba(${l.color}, ${a.toFixed(2)})`;
      this.ctx.lineWidth = 1.4;
      this.ctx.stroke();
      l.age++;
    }
    this.dockLines = this.dockLines.filter(l => l.age < l.max);
  }

  drawDreamOverlay() {
    if (!this.dreaming) return;
    const t = Math.min(1, this.dreamTicks / 30);
    this.ctx.fillStyle = `rgba(20, 30, 60, ${(0.35 * t).toFixed(2)})`;
    this.ctx.fillRect(0, 0, this.width, this.height);
    this.ctx.fillStyle = "rgba(180, 200, 255, 0.7)";
    this.ctx.font = "12px ui-monospace, monospace";
    this.ctx.fillText("dreaming…  flares: " + this.dreamFlares + "/" + this.params.max_downscales_per_layer, 12, this.height - 12);
  }

  drawInset() {
    const cx = this.iw / 2, cy = this.ih / 2;
    const ctx = this.insetCtx;
    ctx.fillStyle = "#050608";
    ctx.fillRect(0, 0, this.iw, this.ih);

    const live = this.triorons.filter(t => t.alive && !t.fading && !t.isDonor);
    const step = 8;
    if (live.length > 0) {
      for (let gy = 0; gy < this.ih; gy += step) {
        for (let gx = 0; gx < this.iw; gx += step) {
          const wx = (gx / this.iw) * this.width;
          const wy = (gy / this.ih) * this.height;
          let nearest = null, ndist = Infinity;
          for (const t of live) {
            const dx = t.x - wx, dy = t.y - wy;
            const d = dx * dx + dy * dy;
            if (d < ndist) { ndist = d; nearest = t; }
          }
          if (nearest) {
            const hue = nearest.effectiveSpecialty(this);
            ctx.fillStyle = `hsla(${hue}, 35%, 45%, 0.7)`;
            ctx.fillRect(gx, gy, step, step);
          }
        }
      }
    }

    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, this.ih);
    ctx.moveTo(0, cy); ctx.lineTo(this.iw, cy);
    ctx.stroke();

    for (const t of live) {
      const ix = (t.x / this.width) * this.iw;
      const iy = (t.y / this.height) * this.ih;
      ctx.beginPath();
      ctx.arc(ix, iy, 2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fill();
    }
  }

  draw() {
    const ctx = this.ctx;
    ctx.fillStyle = "#08090c";
    ctx.fillRect(0, 0, this.width, this.height);

    this.drawAxes();
    for (const p of this.dataPoints) p.draw(ctx);
    this.drawDockLines();
    for (const t of this.triorons) if (t.alive) t.draw(ctx, this);
    this.drawFlashes();
    this.drawDreamOverlay();
    this.drawInset();
  }

  tick() {
    this.tickN++;
    if (this.scene && this.scene.tick) this.scene.tick(this);
    if (this.dreaming) this.tickDream();
    for (const t of this.triorons) t.tick(this);
    for (const p of this.dataPoints) p.tick();
    this.dataPoints = this.dataPoints.filter(p => p.alive);
    this.tickDonors();

    if (!this.dreaming) {
      const thr = this.params.frustration_threshold;
      const toFission = this.triorons.filter(t => t.canFission(thr));
      for (const parent of toFission) this.fission(parent);
    }

    this.triorons = this.triorons.filter(t => t.alive);

    const popEl = document.getElementById("pop-count");
    if (popEl) popEl.textContent = this.triorons.filter(t => !t.isDonor).length;
    const capEl = document.getElementById("pop-cap");
    if (capEl) capEl.textContent = this.params.population_cap;
  }

  loop() {
    // run as many simulation ticks per render frame as world.speed dictates
    this._tickAccum += this.speed;
    let safety = 8;            // cap so a huge speed bump can't lock the page
    while (this._tickAccum >= 1 && safety-- > 0) {
      this.tick();
      this._tickAccum -= 1;
    }
    if (this._tickAccum > 4) this._tickAccum = 4;
    this.draw();
    this._raf = requestAnimationFrame(() => this.loop());
  }
}
