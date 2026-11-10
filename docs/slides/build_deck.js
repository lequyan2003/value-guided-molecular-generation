const pptxgen = require("pptxgenjs");
const path = require("path");
const SK = process.env.SKILL_DIR;
const { applyTheme } = require(SK + "/scripts/apply_theme.js");
const ROOT = path.resolve(__dirname, "../..");
const THEME = { name: "SVDD talk", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2A2F", lt1: "FFFFFF", dk2: "0F3D3E", lt2: "EEF3F2", accent1: "D95F02", accent2: "2A6FBB",
    accent3: "888888", accent4: "0F3D3E", accent5: "7A9E9F", accent6: "B23A48", hlink: "2A6FBB", folHlink: "7A9E9F" } };
const pres = new pptxgen(); pres.layout = "LAYOUT_WIDE"; pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };
pres.title = "Value-guided molecular generation with SVDD on DiGress";
const C = pres.SchemeColor;
pres.defineSlideMaster({ title: "TITLE_ONLY", background: { color: "FFFFFF" },
  slideNumber: { x: 12.5, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: "888888" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontSize: 32, bold: true, color: "0F3D3E", fontFace: "Cambria", margin: 0, align: "left" }, text: "" } }] });
pres.defineSlideMaster({ title: "DARK", background: { color: "0F3D3E" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontSize: 40, bold: true, color: "FFFFFF", fontFace: "Cambria", margin: 0, align: "left" }, text: "" } }] });
const body = (s, items, o) => s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 10 } })),
  Object.assign({ isTextBox: true, fontSize: 22, color: "1B2A2F", valign: "top", margin: 0 }, o));
const note = (s, t) => s.addNotes(t);
const card = (s, x, y, w, h, big, small, col) => {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: "EEF3F2" }, rectRadius: 0.1, line: { color: "EEF3F2" } });
  s.addText(big, { x, y: y + 0.15, w, h: h * 0.5, align: "center", fontSize: 40, bold: true, color: col, fontFace: "Cambria", isTextBox: true, margin: 0 });
  s.addText(small, { x: x + 0.15, y: y + h * 0.58, w: w - 0.3, h: h * 0.38, align: "center", fontSize: 14, color: "1B2A2F", isTextBox: true, margin: 0, valign: "top" });
};

// 1 title
let s = pres.addSlide({ masterName: "DARK" });
s.addText("Value-guided molecular generation", { placeholder: "title" });
s.addText("Soft value-based decoding (SVDD) on a frozen DiGress model, QM9, SA reward", { x: 0.8, y: 4.0, w: 11.5, h: 0.6, fontSize: 22, color: "CADCFC", isTextBox: true, margin: 0 });
s.addText("Abel Le  |  CIS 6930 Generative AI, USF  |  Nov 13, 2026", { x: 0.8, y: 6.3, w: 11.5, h: 0.4, fontSize: 16, color: "FFFFFF", isTextBox: true, margin: 0 });
note(s, "One sentence: can we steer a pretrained graph diffusion model toward molecules that are easier to synthesize, without retraining it, and does it beat just sampling more and keeping the best?");

// 2 problem
s = pres.addSlide({ masterName: "TITLE_ONLY" }); s.addText("Steering a diffusion model without retraining it", { placeholder: "title" });
body(s, ["DiGress generates QM9 molecules as graphs, but samples ignore what we want (e.g. synthesizability, SA score)",
  "Fine-tuning changes the model; best-of-N wastes N full generations",
  "SVDD steers only at inference: at each denoising step, pick among M children using a value estimate",
  "Gap: SVDD was shown on continuous diffusion (GDSS, ZINC). Does it work on discrete graph diffusion, against a fair best-of-N?"],
  { x: 0.6, y: 1.6, w: 7.4, h: 5.2, fontSize: 20 });
card(s, 8.4, 1.8, 4.3, 2.2, "500", "denoising passes for one unguided molecule", "D95F02");
card(s, 8.4, 4.3, 4.3, 2.2, "10", "children per step (M); one DiGress pass serves all M", "2A6FBB");
note(s, "Frozen model: our own training, validity 99.1 %, uniqueness 96.1 %, checkpoint pinned by SHA-256. Official checkpoint link is dead, so we trained it. Best-of-N with N=M matches candidates, not passes.");

// 3 SVDD mechanism
s = pres.addSlide({ masterName: "TITLE_ONLY" }); s.addText("SVDD: propose M children, score, resample", { placeholder: "title" });
const steps = [["1", "Propose", "From state x_t draw M children from the frozen DiGress reverse step"], ["2", "Score", "Value function v(x_{t-1}) estimates the final reward (MLP on the posterior mean, or the posterior-mean reward itself, PM)"], ["3", "Select", "Pick one child with weights proportional to exp(v / alpha): argmax is the alpha-to-0 limit"]];
steps.forEach((st, i) => { const x = 0.6 + i * 4.1;
  s.addShape(pres.ShapeType.roundRect, { x, y: 1.8, w: 3.8, h: 3.2, fill: { color: "EEF3F2" }, rectRadius: 0.1, line: { color: "EEF3F2" } });
  s.addShape(pres.ShapeType.ellipse, { x: x + 0.25, y: 2.0, w: 0.7, h: 0.7, fill: { color: "D95F02" }, line: { color: "D95F02" } });
  s.addText(st[0], { x: x + 0.25, y: 2.0, w: 0.7, h: 0.7, align: "center", valign: "middle", fontSize: 22, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
  s.addText(st[1], { x: x + 1.1, y: 2.0, w: 2.5, h: 0.7, valign: "middle", fontSize: 24, bold: true, color: "0F3D3E", fontFace: "Cambria", isTextBox: true, margin: 0 });
  s.addText(st[2], { x: x + 0.25, y: 2.95, w: 3.3, h: 1.9, fontSize: 20, valign: "top", color: "1B2A2F", isTextBox: true, margin: 0 }); });
s.addText("Cost: unguided T passes; best-of-10 M*T; SVDD-MLP T (children share the pass, values are cheap); SVDD-PM (M+1)*T", { x: 0.6, y: 5.5, w: 12.1, h: 0.9, fontSize: 20, color: "1B2A2F", isTextBox: true, margin: 0 });
note(s, "Explain why children share a pass: DiGress gives a distribution over x_{t-1} from one forward pass, so M samples from it cost one pass. PM needs one extra forward pass per child to get its posterior mean, hence (M+1)T. Derive the soft value: v_t(x) = alpha log E[exp(r/alpha) | x_t = x].");

// 4 setup
s = pres.addSlide({ masterName: "TITLE_ONLY" }); s.addText("Setup: one rule, fixed before running", { placeholder: "title" });
body(s, ["Model: DiGress QM9, trained by us (seed 0, 1000 epochs), frozen, SHA-256 pinned",
  "Reward: SA score (higher = easier to make), 2000 molecules per run, 3 seeds, mean and std over seeds",
  "Baselines: unguided (500 passes), random select, best-of-10 (5000 passes)",
  "Pre-registered rule: SVDD-MLP argmax must beat best-of-10 at equal passes",
  "Metrics: SA reward, strict validity, uniqueness, novelty, diversity, passes per molecule"],
  { x: 0.6, y: 1.6, w: 12.1, h: 4.6 });
note(s, "Stress honesty: the rule was written before results. Softmax over alpha is an exploratory arm added after the pilot showed argmax collapsing. Validity has two definitions: strict (connected, valid) vs RDKit parse; say which.");

// 5 main result
s = pres.addSlide({ masterName: "TITLE_ONLY" }); s.addText("Result: the pre-registered SVDD-MLP rule fails", { placeholder: "title" });
s.addImage({ path: path.join(ROOT, "results/figures/slot_a_headline.png"), x: 0.6, y: 1.5, w: 12.1, h: 3.39, altText: "Bar charts of SA reward, strict validity and uniqueness for eight methods" });
body(s, ["Argmax SVDD-MLP: SA 0.008, validity 0.015 (best-of-10: SA 0.799, validity 1.0)",
  "Softmax a=0.1 / 0.03: SA 0.669 / 0.674 vs 0.635 unguided, validity 0.99 / 0.93 at 500 passes; a=0.01 collapses",
  "SVDD-PM: SA 0.862 but uniqueness 0.54 (reward hacking onto a few molecules)"],
  { x: 0.6, y: 5.1, w: 12.1, h: 1.9, fontSize: 18 });
note(s, "Numbers from tag talk-snapshot, results/tables/summary.csv. Best-of-10 wins the pre-registered comparison. Softmax is cheap (500 passes, 10x fewer than best-of-10) and gives a small, real gain. PM beats best-of-10 on mean SA but at 0.54 uniqueness it collapses diversity.");

// 6 why
s = pres.addSlide({ masterName: "TITLE_ONLY" }); s.addText("Why: the value function is weak and gets exploited", { placeholder: "title" });
s.addImage({ path: path.join(ROOT, "docs/slides/spearman.png"), x: 0.6, y: 1.5, w: 6.4, h: 4.68, altText: "Spearman correlation of the value function with the final reward by timestep" });
body(s, ["Value tracks the final reward late (Spearman 0.6 near t/T=0.2) but fades to 0.08 near pure noise",
  "Ranking siblings of one parent (H1) is weak: Spearman 0.04 to 0.13",
  "Argmax chases value-function errors off the training distribution, so molecules stop being valid",
  "Softmax tempers this: alpha trades reward against staying on the data manifold"],
  { x: 7.4, y: 1.6, w: 5.3, h: 5, fontSize: 20 });
note(s, "H1 (value function ranks siblings) is only weakly supported; H3 is supported. Explain exploitation: the MLP is trained on on-distribution states, argmax pushes into regions it never saw.");

// 7 limits
s = pres.addSlide({ masterName: "TITLE_ONLY" }); s.addText("Limitations", { placeholder: "title" });
body(s, ["One DiGress training seed; 3 sampling seeds",
  "QED only for unguided and best-of-10; SVDD arms are SA only",
  "Several arms ran on one machine; a 5090 replicate of PM and random agrees to about 1e-3",
  "H4 not run; no independent SVDD reimplementation; no real-model M=1 equals unguided test",
  "Softmax arm is exploratory, chosen after seeing the pilot"],
  { x: 0.6, y: 1.6, w: 12.1, h: 4.6 });
note(s, "Say these before they are asked.");

// 8 next
s = pres.addSlide({ masterName: "DARK" }); s.addText("Next: ablations, planned not shown", { placeholder: "title" });
s.addText("M sweep and compute-matched curves  |  GNN value function  |  QED for every arm  |  more seeds  |  H4", { x: 0.8, y: 4.1, w: 11.7, h: 1.0, fontSize: 20, color: "CADCFC", isTextBox: true, margin: 0 });
s.addText("Takeaway: on discrete graph diffusion, naive SVDD does not beat best-of-N; tempered SVDD is a cheap partial win.", { x: 0.8, y: 5.4, w: 11.7, h: 1.0, fontSize: 20, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
note(s, "Demo, if time: scripts/demo.py replay (denoising movie, one SVDD step, head-to-head).");

(async () => { const f = path.join(ROOT, "docs/slides/slides_draft.pptx"); await pres.writeFile({ fileName: f }); await applyTheme(f, THEME); console.log("ok", f); })();
