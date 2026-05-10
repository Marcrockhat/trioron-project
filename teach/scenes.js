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
      "data arrives on the valence axis. each miss adds frustration; " +
      "once a cell crosses the threshold, it fissions. children inherit the same L0 lineage.",
    knob: {
      name: "frustration_threshold", label: "frustration_threshold",
      min: 0.1, max: 1.0, step: 0.05, default: 0.4,
      apiCall: v => `Substrate(frustration_threshold=${v})`,
      lowLabel: "trigger-happy — any miss → fission",
      highLabel: "stoic — many misses needed before splitting",
      mapsTo: "red glow intensity before the yellow fission flash fires",
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
      "stream continues. fissions stack up. the lineage spreads across valence-space, " +
      "each child specializing where it lands. the inset boundary firms up.",
    knob: {
      name: "branch_width", label: "branch_width",
      min: 2, max: 24, step: 1, default: 8,
      apiCall: v => `Substrate.fission(branch_width=${v})`,
      lowLabel: "thin children — many small specialists",
      highLabel: "fat children — fewer, broader cells",
      mapsTo: "initial size of newly-fissioned cells",
    },
    enter(world) {
      if (world.triorons.length === 0) world.spawnGenesis();
    },
    tick(world) {
      runStream(world, 0.9, false, 0);
    },
    exit() {},
  },

  {
    title: "Task 2 — ANGRY vs CALM",
    caption:
      "now the arousal axis. without manifold replay, drift toward T2 erodes T1. " +
      "λ > 0 → faint ghost T1 points re-fire from stored (μ,σ), keeping the valence boundary alive.",
    knob: {
      name: "replay_lambda", label: "replay_lambda",
      min: 0.0, max: 1.0, step: 0.05, default: 0.4,
      apiCall: v => `Substrate.train(replay_lambda=${v.toFixed(2)})`,
      lowLabel: "no replay — T1 boundaries decay as T2 trains",
      highLabel: "strong replay — T1 ghosts keep valence axis alive",
      mapsTo: "rate of faint ghost data points appearing alongside T2",
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
    exit() {},
  },

  {
    title: "First Dream",
    caption:
      "stream pauses. the population stills. nearby cells flare quietly — synaptic " +
      "downscale events, capped per layer. quiet cells with redundant signal fade out.",
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
      if (world.triorons.length === 0) world.spawnGenesis();
      if (world.triorons.filter(t => !t.isDonor).length < 8) {
        world.warmupTo(10, w => runStream(w, 0.9, false, 0));
      }
      world._donorTimer = 30;
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
    exit() {},
  },

  {
    title: "Foreign Donor",
    caption:
      "another donor drifts in — but its inner ring is the wrong lineage. with seed_match=1 " +
      "it bounces off in frustration. flip it to 0 and the forced merge corrupts the host.",
    knob: {
      name: "seed_match_required", label: "seed_match_required",
      min: 0, max: 1, step: 1, default: 1,
      apiCall: v => `Organism.absorb(donor, seed_match_required=${v >= 0.5 ? "True" : "False"})`,
      lowLabel: "0 = False: forced merge → host corruption (specialty desaturates)",
      highLabel: "1 = True: foreign donor bounces off cleanly",
      mapsTo: "red bounce-flash vs orange forced-merge flash",
    },
    enter(world) {
      if (world.triorons.length === 0) world.spawnGenesis();
      if (world.triorons.filter(t => !t.isDonor).length < 8) {
        world.warmupTo(10, w => runStream(w, 0.9, false, 0));
      }
      world._donorTimer = 30;
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
    exit() {},
  },

  {
    title: "Cap Reached — Internal Turnover",
    caption:
      "both streams running hot. once population hits the cap, every new fission " +
      "requires an apoptosis: the quietest cell fades to make room. the deployment regime.",
    knob: {
      name: "population_cap", label: "population_cap",
      min: 30, max: 300, step: 10, default: 200,
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
    },
    tick(world) {
      const phase = Math.floor(world.tickN / 40) % 2;
      runStream(world, 1.0, phase === 1, 0.2);
    },
    exit() {},
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
