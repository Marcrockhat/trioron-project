// Stepper + knob wiring + speed control.
//
// State model:
//   The world is continuous — navigating between scenes does NOT roll back or
//   pause the simulation. Each scene just changes mechanics, caption, and knob.
//   `initialSnap[idx]` is captured the very first time a scene is entered, and
//   the "replay scene" button restores from it (keeping current slider values).
//   `sceneKnobValues[idx]` remembers each scene's slider position separately so
//   `l0_dim` (scene 1) and `replay_lambda` (scene 4) don't interfere.

const dish = document.getElementById("dish");
const inset = document.getElementById("inset");
const world = new World(dish, inset);

const els = {
  chapterNum: document.getElementById("chapter-num"),
  chapterTotal: document.getElementById("chapter-total"),
  chapterTitle: document.getElementById("chapter-title"),
  chapterCaption: document.getElementById("chapter-caption"),
  knob: document.getElementById("knob"),
  knobLabel: document.getElementById("knob-label"),
  knobValue: document.getElementById("knob-value"),
  knobLow: document.getElementById("knob-low"),
  knobHigh: document.getElementById("knob-high"),
  knobMaps: document.getElementById("knob-maps"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  replay: document.getElementById("replay"),
  restart: document.getElementById("restart"),
  apiCall: document.getElementById("api-call"),
  popCap: document.getElementById("pop-cap"),
  speed: document.getElementById("speed"),
  speedVal: document.getElementById("speed-value"),
};

let sceneIdx = 0;
const initialSnap = {};
const sceneKnobValues = {};

function syncSliderFromWorld(scene) {
  const k = scene.knob;
  const v = world.params[k.name] ?? k.default;
  els.knob.min = k.min;
  els.knob.max = k.max;
  els.knob.step = k.step;
  els.knob.value = v;
  els.knobLabel.textContent = k.label;
  if (els.knobLow)  els.knobLow.textContent  = k.lowLabel || "";
  if (els.knobHigh) els.knobHigh.textContent = k.highLabel || "";
  if (els.knobMaps) els.knobMaps.textContent = k.mapsTo || "";
  els.knobValue.textContent = (k.step < 1) ? (+v).toFixed(2) : v;
  els.apiCall.textContent = k.apiCall(+v);
}

function onSliderInput() {
  if (!world.scene) return;
  const k = world.scene.knob;
  const v = +els.knob.value;
  world.params[k.name] = v;
  sceneKnobValues[sceneIdx] = v;
  els.knobValue.textContent = (k.step < 1) ? v.toFixed(2) : v;
  els.apiCall.textContent = k.apiCall(v);
}

function loadScene(idx, { replay = false } = {}) {
  if (world.scene && world.scene.exit) world.scene.exit(world);
  sceneIdx = Math.max(0, Math.min(SCENES.length - 1, idx));
  const scene = SCENES[sceneIdx];

  if (replay) {
    // restart THIS scene from its first-entry state, keeping current slider values
    if (initialSnap[sceneIdx]) {
      const currentParams = { ...world.params };
      world.restore(initialSnap[sceneIdx]);
      world.params = currentParams;
    } else if (sceneIdx === 0) {
      world.reset();
    }
    // Replay = "redo this scene from the beginning" — zero the session-cumulative
    // division counter so the audience sees the count THIS replay produces.
    world.divisionCount = 0;
  }
  // Forward / back navigation: world state continues unchanged across the cut.
  // The audience is moving through chapters of one ongoing story, not loading
  // separate save files.

  // First-ever entry: snapshot the initial state so replay can restore it later.
  if (!initialSnap[sceneIdx]) {
    initialSnap[sceneIdx] = world.snapshot();
  }

  // Directed scenes (scenes 11–13) are controlled experiments that reset the
  // dish. Save the state when entering the directed range; restore it when
  // leaving so the continuous world of scenes 1–10 isn't lost.
  const isDirected = scene.directed === true;
  const wasDirected = world._sandboxActive === true;
  if (isDirected && !wasDirected) {
    world._sandboxBackup = world.snapshot();
    world._sandboxActive = true;
  } else if (!isDirected && wasDirected) {
    if (world._sandboxBackup) {
      const carryParams = { ...world.params };
      world.restore(world._sandboxBackup);
      // Keep the user's tweaked knob values from the rest of the session.
      world.params = carryParams;
      world._sandboxBackup = null;
    }
    world._sandboxActive = false;
  }

  world.scene = scene;
  if (scene.enter) scene.enter(world);

  // Restore the per-scene slider value so each scene "remembers" its own tuning.
  if (sceneKnobValues[sceneIdx] != null) {
    world.params[scene.knob.name] = sceneKnobValues[sceneIdx];
  }

  els.chapterNum.textContent = sceneIdx + 1;
  els.chapterTotal.textContent = SCENES.length;
  els.chapterTitle.textContent = scene.title;
  els.chapterCaption.textContent = scene.caption;
  syncSliderFromWorld(scene);

  if (els.speed) {
    const sliderMax = +els.speed.max;
    els.speed.value = Math.min(world.speed, sliderMax);
    if (els.speedVal) els.speedVal.textContent = world.speed.toFixed(2) + "×";
  }

  els.prev.disabled = sceneIdx === 0;
  els.next.disabled = sceneIdx === SCENES.length - 1;
}

if (els.speed) {
  els.speed.addEventListener("input", () => {
    const v = +els.speed.value;
    world.speed = v;
    if (els.speedVal) els.speedVal.textContent = v.toFixed(2) + "×";
  });
}

els.knob.addEventListener("input", onSliderInput);
els.next.addEventListener("click", () => loadScene(sceneIdx + 1));
els.prev.addEventListener("click", () => loadScene(sceneIdx - 1));
els.replay.addEventListener("click", () => loadScene(sceneIdx, { replay: true }));
if (els.restart) {
  els.restart.addEventListener("click", () => {
    // Hard reset: wipe world, all per-scene snapshots, all remembered slider
    // values, and any sandbox state — then load scene 1 from scratch.
    world.reset();
    world._sandboxBackup = null;
    world._sandboxActive = false;
    for (const k of Object.keys(initialSnap)) delete initialSnap[k];
    for (const k of Object.keys(sceneKnobValues)) delete sceneKnobValues[k];
    loadScene(0);
  });
}

els.popCap.textContent = world.params.population_cap;

loadScene(0);
world.loop();
