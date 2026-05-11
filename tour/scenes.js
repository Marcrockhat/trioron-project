// Scene definitions. Each scene has enter/tick/exit hooks and a single knob
// that maps to a real trioron API parameter. Each knob also carries readable
// info on what low/high mean and which visual feature it controls.

const SCENES = [
  {
    title: "Genesis",
    caption: "this is the substrate. it knows nothing yet. one cell, frozen L0, no branches.",
    knob: {
      name: "l0_dim", label: "l0_dim",
      min: 1, max: 32, step: 1, default: 8,
      apiCall: v => `trioron.Substrate(l0_dim=${v})`,
      lowLabel: "narrow substrate — can't carry fine distinctions",
      highLabel: "wide substrate — room to specialize later",
      mapsTo: "outer body radius of the cell",
    },
    enter(world) {
      // non-destructive: only spawn if the world is genuinely empty.
      // Going back to scene 1 from scene 2 will already have restored
      // an existing cell; we leave it alone.
      if (world.triorons.length === 0) world.spawnGenesis();
    },
    tick(world) {
      // l0_dim controls every cell's inner-ring (L0 substrate) width — they all
      // share the substrate, so we update them together.
      for (const t of world.triorons) {
        if (!t.isDonor) t.substrateWidth = world.params.l0_dim;
      }
    },
    exit() {},
  },

  {
    title: "First Task — HAPPY vs SAD",
    caption:
      "data arrives on the valence axis. each miss adds frustration; once a cell crosses " +
      "the threshold, it divides. children inherit the same L0 lineage. counter-intuitively, " +
      "a LOW threshold can produce FEWER total divisions: the first division fires while the " +
      "parent is still near centre, and its random-angle child has ~50% odds of landing in " +
      "the unmet specialty — the task resolves in one shot. a HIGH threshold delays the " +
      "first division until the parent has drifted, so children may land in already-covered " +
      "regions, triggering more divisions before resolution. press replay to see the variance; " +
      "the division counter above tracks how many actually fired.",
    knob: {
      name: "frustration_threshold", label: "frustration_threshold",
      min: 0.1, max: 1.0, step: 0.05, default: 0.4,
      apiCall: v => `Substrate(frustration_threshold=${v})`,
      lowLabel: "trigger-happy — any miss → division",
      highLabel: "stoic — many misses needed before splitting",
      mapsTo: "red glow intensity before the yellow division flash fires",
    },
    enter(world) {
      if (world.triorons.length === 0) world.spawnGenesis();
    },
    tick(world) {
      runStream(world, 0.6, /* t2 */ false, /* replay */ 0);
    },
    exit() {},
  },

  {
    title: "Population Grows",
    caption:
      "stream continues. misses build frustration; threshold-crossing cells divide and " +
      "their children specialize where they land. over enough ticks the lineage tends to " +
      "spread across both halves of valence-space and the dashed boundary tends to firm up — " +
      "but the exact number of divisions and the path depends on data order and where " +
      "children land. division counter (top) tracks the actual count.",
    knob: {
      name: "branch_width", label: "branch_width",
      min: 2, max: 24, step: 1, default: 8,
      apiCall: v => `Substrate.divide(branch_width=${v})`,
      lowLabel: "thin children — many small specialists",
      highLabel: "fat children — fewer, broader cells",
      mapsTo: "outer body size of every branched cell (live retro-applied)",
    },
    enter(world) {
      if (world.triorons.length === 0) world.spawnGenesis();
    },
    tick(world) {
      // Live retro-apply: existing branched cells inherit the slider value so the
      // knob has immediate visual feedback. Pure-substrate cells (branchWidth=0)
      // are left alone so the L0-only genesis state still reads on a revisit.
      for (const t of world.triorons) {
        if (!t.isDonor && !t.fading && t.branchWidth > 0) {
          t.branchWidth = world.params.branch_width;
        }
      }
      runStream(world, 0.9, false, 0);
    },
    drawOverlay(world) { world.drawValenceBoundary(); },
    exit() {},
  },

  {
    title: "Task 2 — ANGRY vs CALM",
    caption:
      "now the arousal axis. without manifold replay (λ=0), new divisions tend to migrate " +
      "toward arousal corners and the valence boundary tends to thin over many ticks. " +
      "λ > 0 → faint ghost T1 points re-fire from stored (μ,σ), and the valence boundary " +
      "tends to stay thick and sharp. trend reliable; exact pace varies per run.",
    knob: {
      name: "replay_lambda", label: "replay_lambda",
      min: 0.0, max: 1.0, step: 0.05, default: 0.4,
      apiCall: v => `Substrate.train(replay_lambda=${v.toFixed(2)})`,
      lowLabel: "no replay — valence boundary thins as T2 trains",
      highLabel: "strong replay — boundary stays thick and saturated",
      mapsTo: "thickness/opacity of the dashed valence boundary line",
    },
    enter(world) {
      if (world.triorons.length === 0) world.spawnGenesis();
      // give the audience a baseline if pop is too thin
      if (world.triorons.filter(t => !t.isDonor).length < 6) {
        world.warmupTo(8, w => runStream(w, 0.9, false, 0));
      }
    },
    tick(world) {
      runStream(world, 0.7, /* t2 */ true, world.params.replay_lambda);
    },
    drawOverlay(world) { world.drawValenceBoundary(); },
    exit() {},
  },

  {
    title: "First Dream",
    caption:
      "stream pauses. nearby cells flare quietly — synaptic downscale events, capped per " +
      "layer. quiet cells with redundant signal tend to fade; centre cells usually go first " +
      "since they sit between specialties and fire least. exact targets depend on which " +
      "cells happened to fire least this run. the L0 inner ring lingers as a ghost: " +
      "substrate is conserved, only the L1 branch is pruned.",
    knob: {
      name: "max_downscales_per_layer", label: "max_downscales_per_layer",
      min: 1, max: 24, step: 1, default: 8,
      apiCall: v => `Substrate.dream(max_downscales_per_layer=${v})`,
      lowLabel: "few flares — minimal consolidation",
      highLabel: "many flares — aggressive prune of redundancy",
      mapsTo: "blue line-flares between cell pairs during dream phase",
    },
    enter(world) {
      if (world.triorons.length === 0) world.spawnGenesis();
      if (world.triorons.filter(t => !t.isDonor).length < 8) {
        world.warmupTo(10, w => runStream(w, 0.9, false, 0));
      }
      world.startDream();
    },
    tick(world) {
      // dream auto-ends after ~150 ticks; restart so the audience can re-tune the knob
      if (!world.dreaming && world.tickN % 80 === 0) world.startDream();
    },
    exit(world) {
      world.dreaming = false;
    },
  },

  {
    title: "Compatible Donor",
    caption:
      "a foreign-trained cell drifts in. its inner ring matches the lineage — same L0 seed. " +
      "within absorption_radius, it docks; the host gains its branch. lossless paste-and-go.",
    knob: {
      name: "absorption_radius", label: "absorption_radius",
      min: 20, max: 160, step: 5, default: 60,
      apiCall: v => `Organism.absorb(donor, absorption_radius=${v})`,
      lowLabel: "tight — donors must approach head-on",
      highLabel: "loose — long-range capture",
      mapsTo: "dashed circle around each donor — reach",
    },
    enter(world) {
      world.clearDonors();
      if (world.triorons.length === 0) world.spawnGenesis();
      if (world.triorons.filter(t => !t.isDonor).length < 8) {
        world.warmupTo(10, w => runStream(w, 0.9, false, 0));
      }
      // Spawn the first compatible donor immediately so the dashed
      // absorption_radius circle is on-screen from frame 1.
      world.spawnDonor({ seedColor: LINEAGE_HOME, specialty: Math.random() * 360 });
      world._donorTimer = 240;
    },
    tick(world) {
      runStream(world, 0.2, world.tickN % 60 < 30, 0.1);
      world._donorTimer--;
      if (world._donorTimer <= 0) {
        const liveDonors = world.triorons.filter(t => t.isDonor && !t.docked && !t.fading);
        if (liveDonors.length < 2) {
          const specialty = Math.random() * 360;
          world.spawnDonor({ seedColor: LINEAGE_HOME, specialty });
        }
        world._donorTimer = 200;
      }
    },
    exit(world) { world.clearDonors(); },
  },

  {
    title: "Foreign Donor",
    caption:
      "another donor drifts in — its inner ring tags a different training pool (different " +
      "shared factor S). seed mismatch alone is solved by the 4-byte handshake (W = R·S, " +
      "lossless), but pool mismatch is the deep one: the L1 branches were shaped against a " +
      "different S. with pool_match=1 it bounces off cleanly; flip to 0 and the forced merge " +
      "corrupts the host's specialty.",
    knob: {
      name: "pool_match_required", label: "pool_match_required",
      min: 0, max: 1, step: 1, default: 1,
      apiCall: v => `Organism.absorb(donor, pool_match_required=${v >= 0.5 ? "True" : "False"})`,
      lowLabel: "0 = False: forced merge → host corruption (specialty desaturates)",
      highLabel: "1 = True: pool-mismatched donor bounces off cleanly",
      mapsTo: "red bounce-flash vs orange forced-merge flash",
    },
    enter(world) {
      world.clearDonors();
      // Reset absorption_radius so a value the user cranked up in scene 6 doesn't
      // make the foreign donor bounce mid-flight before the trajectory is visible.
      world.params.absorption_radius = 60;
      if (world.triorons.length === 0) world.spawnGenesis();
      if (world.triorons.filter(t => !t.isDonor).length < 8) {
        world.warmupTo(10, w => runStream(w, 0.9, false, 0));
      }
      // Spawn the first foreign donor immediately so the bounce / forced-merge
      // is observable as soon as the audience reads the caption.
      world.spawnDonor({ seedColor: LINEAGE_FOREIGN, specialty: Math.random() * 360 });
      world._donorTimer = 240;
    },
    tick(world) {
      runStream(world, 0.2, world.tickN % 60 < 30, 0.1);
      world._donorTimer--;
      if (world._donorTimer <= 0) {
        const liveDonors = world.triorons.filter(t => t.isDonor && !t.docked && !t.fading);
        if (liveDonors.length < 2) {
          const specialty = Math.random() * 360;
          world.spawnDonor({ seedColor: LINEAGE_FOREIGN, specialty });
        }
        world._donorTimer = 220;
      }
    },
    exit(world) { world.clearDonors(); },
  },

  {
    title: "Cap Reached — Internal Turnover",
    caption:
      "both streams running hot. once population hits the cap, every new division " +
      "requires an apoptosis: the quietest cell fades to make room. the deployment regime. " +
      "the simulation runs accelerated on this chapter so the cap is reached in viewing time; " +
      "drag the speed slider down if you want to watch a single turnover step.",
    knob: {
      name: "population_cap", label: "population_cap",
      min: 30, max: 200, step: 10, default: 200,
      apiCall: v => `Organism(population_cap=${v})`,
      lowLabel: "tight budget — heavy turnover, crowded apoptosis",
      highLabel: "loose budget — population grows freely without cannibalization",
      mapsTo: "ceiling shown in the population readout (top-left)",
    },
    enter(world) {
      if (world.triorons.length === 0) world.spawnGenesis();
      if (world.triorons.filter(t => !t.isDonor).length < 10) {
        world.warmupTo(15, w => runStream(w, 0.9, false, 0));
      }
      world.speed = 8;
    },
    tick(world) {
      const phase = Math.floor(world.tickN / 40) % 2;
      runStream(world, 1.0, phase === 1, 0.2);
    },
    exit(world) { world.speed = 1; },
  },

  {
    title: "Replay & TTS — click anywhere",
    caption:
      "this is the population that emerged. click on the canvas — the nearest emotion's clip " +
      "plays. that's what trioron would condition on the device. drop the σ-scale to 0 to watch decay.",
    knob: {
      name: "manifold_noise_scale", label: "manifold_noise_scale",
      min: 0.0, max: 2.0, step: 0.1, default: 1.0,
      apiCall: v => `Substrate.replay(manifold_noise_scale=${v.toFixed(1)})`,
      lowLabel: "0 = means-only — brittle, decays under T2",
      highLabel: "wide σ — rich per-class signatures, robust",
      mapsTo: "density of ghost data points re-firing each tick",
    },
    enter(world) {
      if (world.triorons.length === 0) world.spawnGenesis();
      if (world.triorons.filter(t => !t.isDonor).length < 12) {
        world.warmupTo(18, w => runStream(w, 0.9, false, 0));
      }
      world.onClick = (x, y) => playEmotionAt(world, x, y);
    },
    tick(world) {
      const lam = Math.min(1.0, world.params.manifold_noise_scale * 0.4);
      runStream(world, 0.6, true, lam);
    },
    exit(world) {
      world.onClick = null;
    },
  },

  {
    title: "Expansion — DISGUST + PRIDE",
    caption:
      "the organism is done — and a new task arrives. api.extend() opens two new labels " +
      "in the empty upper corners; divisions grow into the new region. with replay during " +
      "extend, the valence boundary tends to stay sharp; without it, the old skill tends " +
      "to thin as the population pivots. trend reliable; runs vary in pace.",
    knob: {
      name: "extend_replay_lambda", label: "extend_replay_lambda",
      min: 0.0, max: 1.0, step: 0.05, default: 0.5,
      apiCall: v =>
        `api.extend(new_labels=["DISGUST","PRIDE"], replay_lambda=${v.toFixed(2)})`,
      lowLabel: "0 = no replay — old valence skill forgets",
      highLabel: "1 = strong replay — lossless expansion",
      mapsTo: "thickness of the dashed valence boundary as new data streams in",
    },
    enter(world) {
      if (world.triorons.length === 0) world.spawnGenesis();
      if (world.triorons.filter(t => !t.isDonor).length < 12) {
        world.warmupTo(20, w => runStream(w, 0.9, false, 0));
      }
      world.onClick = (x, y) => playEmotionAt(world, x, y);
    },
    tick(world) {
      const lam = world.params.extend_replay_lambda;
      // new-task stream: DISGUST + PRIDE arrive in upper corners
      if (Math.random() < 0.65) spawnT3Point(world, false);
      // replay of all previously-learned tasks (T1 valence + T2 arousal) as faint ghosts
      if (lam > 0) {
        if (Math.random() < lam * 0.45) spawnT1Point(world, true);
        if (Math.random() < lam * 0.45) spawnT2Point(world, true);
      }
      for (const p of world.dataPoints) {
        if (!p.fadeColor && p.age > 8) world.classifyPoint(p);
      }
    },
    drawOverlay(world) { world.drawValenceBoundary(); },
    exit(world) {
      world.onClick = null;
    },
  },

  // ---------------------- directed cases (11–13) ----------------------
  // Three preconfigured scenarios that demonstrate concrete outcomes the
  // architecture supports: clean convergence, failure to converge, and
  // catastrophic forgetting on expansion. Each scene preloads its world.params
  // on enter() so the audience can see the textbook case immediately; the
  // single per-scene knob lets them escape (or deepen) the demonstration.
  // The drawConvergenceMetric() readout in the top-left is the unifying signal.

  {
    directed: true,
    title: "Directed: Convergence",
    caption:
      "controlled experiment — dish resets on entry. balanced HAPPY/SAD/ANGRY/CALM stream " +
      "with replay on; cap=80; frustration_threshold at sweet spot. specialists tend to fill " +
      "all four quadrants and the Σ frustration readout tends to drop over time. crank the " +
      "knob up to stall division (Σ plateaus), drop it to see division-spam.",
    knob: {
      name: "frustration_threshold", label: "frustration_threshold",
      min: 0.1, max: 1.0, step: 0.05, default: 0.4,
      apiCall: v => `Substrate(frustration_threshold=${v})`,
      lowLabel: "low — division-spam, unstable specialists",
      highLabel: "high — division stalls, frustration plateaus",
      mapsTo: "Σ frustration readout (top-left)",
    },
    enter(world) {
      world.reset();
      world.dreaming = false;
      world.params.replay_lambda = 0.5;
      world.params.population_cap = 80;
      world.params.frustration_threshold = 0.4;
      world.params.min_live_pop = 5;                // default — undo any prior scene's override
      world.spawnGenesis();
      // Seed a modest, frustration-charged starter pop so visible division begins
      // immediately rather than waiting through the first few misclassifications.
      world.warmupTo(6, w => runStream(w, 0.9, false, 0));
    },
    tick(world) {
      const phase = Math.floor(world.tickN / 40) % 2;
      runStream(world, 0.8, phase === 1, world.params.replay_lambda);
    },
    drawOverlay(world) {
      world.drawValenceBoundary();
      world.drawConvergenceMetric();
    },
    exit() {},
  },

  {
    directed: true,
    title: "Directed: Fails to Converge",
    caption:
      "controlled experiment — dish resets on entry. cap is locked at 3 and the dream " +
      "floor (min_live_pop) is dropped to 2, so the population genuinely caps at 3 cells " +
      "for 4 data quadrants. at least one quadrant is always uncovered → frustration " +
      "stays high, divisions can't escape the cap, ghost rings pile up from apoptosis. " +
      "raise the cap above 4 to let the population cover all four corners.",
    knob: {
      name: "population_cap", label: "population_cap",
      min: 2, max: 80, step: 1, default: 3,
      apiCall: v => `Organism(population_cap=${v})`,
      lowLabel: "tight — under-capacity, no stable specialists",
      highLabel: "loose — enough room for all four quadrants",
      mapsTo: "Σ frustration readout — should drop as cap rises above 4",
    },
    enter(world) {
      world.reset();
      world.dreaming = false;
      world.params.replay_lambda = 0.3;
      world.params.frustration_threshold = 0.35;
      world.params.population_cap = 3;
      world.params.min_live_pop = 2;            // lower the floor so the cap actually bites
      world.spawnGenesis();
      // Warm up to exactly the cap so divisions can't grow past it from the start.
      world.warmupTo(3, w => runStream(w, 0.9, false, 0));
    },
    tick(world) {
      const phase = Math.floor(world.tickN / 30) % 2;
      runStream(world, 0.95, phase === 1, world.params.replay_lambda);
    },
    drawOverlay(world) {
      world.drawValenceBoundary();
      world.drawConvergenceMetric();
    },
    exit() {},
  },

  {
    directed: true,
    title: "Directed: Expansion Forgets",
    caption:
      "controlled experiment — dish resets on entry. a population is pre-trained on " +
      "HAPPY/SAD/ANGRY/CALM, then api.extend() opens DISGUST + PRIDE with replay locked " +
      "at 0. the valence boundary tends to thin as divisions chase upper-corner data " +
      "and the old skill tends to fade. lift the knob to restore replay and both axes " +
      "tend to survive.",
    knob: {
      name: "extend_replay_lambda", label: "extend_replay_lambda",
      min: 0.0, max: 1.0, step: 0.05, default: 0.0,
      apiCall: v =>
        `api.extend(new_labels=["DISGUST","PRIDE"], replay_lambda=${v.toFixed(2)})`,
      lowLabel: "0 — old valence skill forgotten",
      highLabel: "1 — lossless expansion, both old and new survive",
      mapsTo: "valence boundary line — thin = forgotten, thick = retained",
    },
    enter(world) {
      world.reset();
      world.dreaming = false;
      world.params.population_cap = 80;
      world.params.frustration_threshold = 0.4;
      world.params.min_live_pop = 5;                // default — undo any prior scene's override
      world.spawnGenesis();
      // Pre-train on T1 + T2 so there's an *old* skill that can be forgotten.
      world.warmupTo(16, w => {
        const phase = Math.floor(w.tickN / 30) % 2;
        runStream(w, 0.9, phase === 1, 0.4);
      });
    },
    tick(world) {
      const lam = world.params.extend_replay_lambda;
      if (Math.random() < 0.75) spawnT3Point(world, false);
      if (lam > 0) {
        if (Math.random() < lam * 0.45) spawnT1Point(world, true);
        if (Math.random() < lam * 0.45) spawnT2Point(world, true);
      }
      for (const p of world.dataPoints) {
        if (!p.fadeColor && p.age > 8) world.classifyPoint(p);
      }
    },
    drawOverlay(world) {
      world.drawValenceBoundary();
      world.drawConvergenceMetric();
    },
    exit() {},
  },
];

// ---------- closing-beat audio ----------
//
// Contrast notes: Web Speech API's prosody is limited, so we push rate / pitch
// to their extremes and stack a volume envelope on top. A longer phrase gives
// the prosody more room to register. Drop real RAVDESS clips into audio/<key>.mp3
// for the audible upgrade.

const EMOTION_PHRASE = "the world is what it is, and so are we.";

const EMOTION_AUDIO = [
  // hue 0 ↔ angry (red), 50 ↔ happy (yellow), 130 ↔ calm (green),
  // 180 ↔ tender (cyan), 230 ↔ sad (blue), 270 ↔ bored (purple),
  // 320 ↔ anxious (magenta).
  { key: "happy",    hue: 50,  rate: 1.55, pitch: 1.7,  volume: 1.0 },
  { key: "excited",  hue: 25,  rate: 1.85, pitch: 1.55, volume: 1.0 },
  { key: "angry",    hue: 0,   rate: 1.6,  pitch: 0.55, volume: 1.0 },
  { key: "anxious",  hue: 320, rate: 1.5,  pitch: 1.85, volume: 0.85 },
  { key: "sad",      hue: 230, rate: 0.55, pitch: 0.45, volume: 0.7 },
  { key: "bored",    hue: 270, rate: 0.6,  pitch: 0.55, volume: 0.55 },
  { key: "calm",     hue: 130, rate: 0.75, pitch: 0.95, volume: 0.8 },
  { key: "tender",   hue: 180, rate: 0.85, pitch: 1.35, volume: 0.7 },
  // scene 10 expansion labels
  { key: "disgust",  hue: 290, rate: 0.7,  pitch: 0.6,  volume: 0.65 },
  { key: "pride",    hue: 20,  rate: 1.35, pitch: 1.15, volume: 1.0 },
];

let _voiceCache = null;
function pickVoice() {
  if (typeof speechSynthesis === "undefined") return null;
  if (_voiceCache) return _voiceCache;
  const voices = speechSynthesis.getVoices();
  if (!voices || voices.length === 0) return null;
  // prefer a non-default English voice; some browsers expose more expressive ones
  const en = voices.filter(v => /en[-_]?/i.test(v.lang));
  const preferred = en.find(v => /female|samantha|karen|moira|tessa|google.*us english/i.test(v.name));
  _voiceCache = preferred || en[0] || voices[0];
  return _voiceCache;
}
if (typeof speechSynthesis !== "undefined") {
  // voices load asynchronously; refresh cache when they arrive
  speechSynthesis.addEventListener?.("voiceschanged", () => { _voiceCache = null; pickVoice(); });
}

function playEmotionAt(world, x, y) {
  const hue = positionToHue(world, x, y);
  let best = null, bd = Infinity;
  for (const e of EMOTION_AUDIO) {
    const d = hueDist(hue, e.hue);
    if (d < bd) { bd = d; best = e; }
  }
  if (!best) return;

  const audioPath = `audio/${best.key}.mp3`;
  const a = new Audio(audioPath);
  a.volume = best.volume;
  a.play().catch(() => {
    if (typeof speechSynthesis !== "undefined") {
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(EMOTION_PHRASE);
      u.rate = best.rate;
      u.pitch = best.pitch;
      u.volume = best.volume;
      const v = pickVoice();
      if (v) u.voice = v;
      speechSynthesis.speak(u);
    }
  });

  // visual ping at the click site, tinted by the matched emotion
  world.flashes.push({ x, y, age: 0, max: 36, color: "246, 201, 113" });

  // also flash the chapter caption with which emotion was matched
  const cap = document.getElementById("chapter-caption");
  if (cap) {
    cap.dataset.lastEmotion = best.key;
    cap.style.borderColor = `hsl(${best.hue}, 70%, 60%)`;
    setTimeout(() => { cap.style.borderColor = ""; }, 900);
  }
}
