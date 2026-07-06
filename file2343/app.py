"""
app.py  –  ASD Behavior Detection · Flask Web App
Backend is unchanged. Only the HTML/CSS/JS template is redesigned.
"""

import os, uuid, json
from flask import Flask, request, jsonify, render_template_string
from inference import BehaviorDetector, CLASSES, COLORS, NO_BEHAVIOR_LABEL, NO_BEHAVIOR_COLOR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector = BehaviorDetector(model_path="lstm_model.pth")

# ────────────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuroScan · ASD Behavior Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ═══════════════════════════════════════════════════
   TOKENS
═══════════════════════════════════════════════════ */
:root {
  --bg0:        #04070f;
  --bg1:        #080f1e;
  --bg2:        #0d1630;
  --glass:      rgba(255,255,255,0.042);
  --glass-b:    rgba(255,255,255,0.07);
  --glow-p:     #7c3aed;
  --glow-c:     #06b6d4;
  --glow-g:     #10b981;
  --accent:     #818cf8;
  --accent2:    #38bdf8;
  --text:       #f0f4ff;
  --text2:      #94a3b8;
  --text3:      #475569;
  --arm:        #f43f5e;
  --head:       #fb923c;
  --spin:       #38bdf8;
  --ok:         #34d399;
  --border:     rgba(255,255,255,0.07);
  --r:          16px;
  --r-sm:       10px;
  --shadow:     0 8px 40px rgba(0,0,0,0.55);
  --font:       'Space Grotesk', sans-serif;
  --mono:       'JetBrains Mono', monospace;
}

/* ═══════════════════════════════════════════════════
   RESET & BASE
═══════════════════════════════════════════════════ */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  font-family:var(--font);
  background:var(--bg0);
  color:var(--text);
  min-height:100vh;
  overflow-x:hidden;
}

/* ═══════════════════════════════════════════════════
   ANIMATED CANVAS BACKGROUND
═══════════════════════════════════════════════════ */
#neural-bg{
  position:fixed;inset:0;z-index:0;
  pointer-events:none;
}
.page-wrap{
  position:relative;z-index:1;
  max-width:900px;
  margin:0 auto;
  padding:0 20px 100px;
}

/* ═══════════════════════════════════════════════════
   GRADIENT ORBS  (ambient glow)
═══════════════════════════════════════════════════ */
.orb{
  position:fixed;border-radius:50%;filter:blur(100px);
  pointer-events:none;z-index:0;opacity:0.18;
  animation:orb-drift 18s ease-in-out infinite alternate;
}
.orb-1{width:600px;height:600px;background:var(--glow-p);top:-150px;left:-180px;animation-delay:0s}
.orb-2{width:500px;height:500px;background:var(--glow-c);top:30%;right:-160px;animation-delay:-6s}
.orb-3{width:400px;height:400px;background:var(--glow-g);bottom:-100px;left:30%;animation-delay:-12s;opacity:0.10}
@keyframes orb-drift{
  from{transform:translate(0,0) scale(1)}
  to  {transform:translate(30px,20px) scale(1.08)}
}

/* ═══════════════════════════════════════════════════
   HEADER
═══════════════════════════════════════════════════ */
.hero{
  padding:72px 0 56px;
  display:flex;flex-direction:column;align-items:flex-start;
  animation:fade-up 0.8s ease both;
}
.hero-chip{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(124,58,237,0.15);
  border:1px solid rgba(124,58,237,0.35);
  color:#a78bfa;
  font-family:var(--mono);font-size:11px;letter-spacing:1.5px;
  padding:5px 14px;border-radius:999px;
  text-transform:uppercase;margin-bottom:24px;
}
.hero-chip::before{
  content:'';width:7px;height:7px;border-radius:50%;
  background:#a78bfa;
  animation:pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot{
  0%,100%{opacity:1;transform:scale(1)}
  50%{opacity:0.4;transform:scale(0.7)}
}
.hero h1{
  font-size:clamp(36px,6vw,64px);
  font-weight:700;line-height:1.08;letter-spacing:-1.5px;
  color:var(--text);
}
.hero h1 .grad{
  background:linear-gradient(135deg,#818cf8 0%,#38bdf8 50%,#34d399 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}
.hero-sub{
  margin-top:18px;max-width:580px;
  color:var(--text2);font-size:16px;line-height:1.65;font-weight:400;
}
.hero-meta{
  margin-top:20px;display:flex;gap:20px;flex-wrap:wrap;
}
.meta-pill{
  display:flex;align-items:center;gap:6px;
  font-family:var(--mono);font-size:11px;color:var(--text3);
  border:1px solid var(--border);border-radius:999px;
  padding:4px 12px;
}
.meta-pill span{color:var(--accent);font-weight:500}

/* ═══════════════════════════════════════════════════
   GLASS CARD
═══════════════════════════════════════════════════ */
.card{
  background:var(--glass);
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  border:1px solid var(--border);
  border-radius:var(--r);
  padding:32px;
  margin-bottom:20px;
  box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,0.05);
  animation:fade-up 0.6s ease both;
}
.card:nth-child(1){animation-delay:0.1s}
.card:nth-child(2){animation-delay:0.2s}
.card:nth-child(3){animation-delay:0.3s}
@keyframes fade-up{
  from{opacity:0;transform:translateY(24px)}
  to  {opacity:1;transform:translateY(0)}
}

/* card section label */
.section-label{
  font-family:var(--mono);font-size:10px;letter-spacing:2px;
  text-transform:uppercase;color:var(--text3);
  margin-bottom:20px;display:flex;align-items:center;gap:8px;
}
.section-label::after{
  content:'';flex:1;height:1px;background:var(--border);
}

/* ═══════════════════════════════════════════════════
   UPLOAD ZONE
═══════════════════════════════════════════════════ */
.drop-zone{
  position:relative;
  border:2px dashed rgba(255,255,255,0.10);
  border-radius:var(--r-sm);
  padding:52px 24px;
  text-align:center;
  cursor:pointer;
  transition:border-color .25s,background .25s,transform .2s;
  overflow:hidden;
}
.drop-zone::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 50% 0%,rgba(99,102,241,0.06) 0%,transparent 70%);
  opacity:0;transition:opacity .3s;
}
.drop-zone:hover::before,.drop-zone.drag-over::before{opacity:1}
.drop-zone:hover,.drop-zone.drag-over{
  border-color:rgba(129,140,248,0.5);
  background:rgba(99,102,241,0.04);
  transform:scale(1.005);
}
.drop-zone input[type="file"]{
  position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;
}
.dz-icon{
  width:64px;height:64px;margin:0 auto 18px;
  background:linear-gradient(135deg,rgba(99,102,241,0.2),rgba(56,189,248,0.2));
  border:1px solid rgba(129,140,248,0.25);
  border-radius:50%;display:flex;align-items:center;justify-content:center;
}
.dz-icon svg{width:28px;height:28px}
.dz-title{font-size:15px;font-weight:600;color:var(--text);margin-bottom:6px}
.dz-hint{font-size:13px;color:var(--text2)}
.dz-hint strong{color:var(--accent)}
.dz-formats{
  margin-top:8px;font-family:var(--mono);font-size:11px;color:var(--text3);
}
.dz-selected{
  margin-top:14px;
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.25);
  border-radius:8px;padding:7px 14px;
  font-family:var(--mono);font-size:12px;color:var(--ok);
  animation:pop-in .3s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes pop-in{
  from{opacity:0;transform:scale(0.85)}
  to{opacity:1;transform:scale(1)}
}

/* ═══════════════════════════════════════════════════
   ANALYZE BUTTON
═══════════════════════════════════════════════════ */
.btn-analyze{
  width:100%;margin-top:20px;
  padding:15px 24px;
  background:linear-gradient(135deg,#6366f1,#38bdf8);
  color:#fff;font-family:var(--font);font-size:15px;font-weight:600;
  border:none;border-radius:var(--r-sm);
  cursor:pointer;
  position:relative;overflow:hidden;
  transition:opacity .2s,transform .15s,box-shadow .2s;
  box-shadow:0 4px 24px rgba(99,102,241,0.35);
  letter-spacing:0.3px;
}
.btn-analyze::after{
  content:'';position:absolute;
  top:-50%;left:-60%;
  width:40%;height:200%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.25),transparent);
  transform:skewX(-15deg);
  animation:shine 3.5s ease-in-out infinite;
}
@keyframes shine{
  0%{left:-60%}60%,100%{left:130%}
}
.btn-analyze:hover:not(:disabled){
  box-shadow:0 6px 32px rgba(99,102,241,0.55);transform:translateY(-1px);
}
.btn-analyze:active:not(:disabled){transform:translateY(0) scale(0.99)}
.btn-analyze:disabled{opacity:0.35;cursor:not-allowed}
.btn-analyze:disabled::after{animation:none}

/* ═══════════════════════════════════════════════════
   PROGRESS
═══════════════════════════════════════════════════ */
.progress-wrap{display:none;margin-top:20px}
.progress-steps{
  display:flex;gap:0;margin-bottom:14px;
  border-radius:8px;overflow:hidden;
  border:1px solid var(--border);
}
.p-step{
  flex:1;padding:9px 6px;text-align:center;
  font-family:var(--mono);font-size:10px;color:var(--text3);
  border-right:1px solid var(--border);
  transition:background .4s,color .4s;
}
.p-step:last-child{border-right:none}
.p-step.active{background:rgba(99,102,241,0.15);color:var(--accent)}
.p-step.done{background:rgba(52,211,153,0.08);color:var(--ok)}
.progress-track{
  height:3px;background:rgba(255,255,255,0.06);
  border-radius:999px;overflow:hidden;
}
.progress-fill{
  height:100%;width:0%;
  background:linear-gradient(90deg,#6366f1,#38bdf8,#34d399);
  border-radius:999px;
  transition:width .5s cubic-bezier(0.4,0,0.2,1);
  box-shadow:0 0 12px rgba(99,102,241,0.6);
}
.scan-label{
  margin-top:8px;font-family:var(--mono);font-size:11px;
  color:var(--text3);text-align:right;
}

/* ═══════════════════════════════════════════════════
   ERROR
═══════════════════════════════════════════════════ */
.error-box{
  display:none;margin-top:14px;
  background:rgba(244,63,94,0.08);
  border:1px solid rgba(244,63,94,0.30);
  border-radius:var(--r-sm);
  padding:14px 18px;
  color:#fb7185;font-size:13px;
  animation:fade-up .3s ease;
}

/* ═══════════════════════════════════════════════════
   RESULT CARD  (hidden until result)
═══════════════════════════════════════════════════ */
#result-card{display:none}

/* slide-in animation when result arrives */
.slide-in{
  animation:slide-in-up .55s cubic-bezier(0.16,1,0.3,1) both;
}
@keyframes slide-in-up{
  from{opacity:0;transform:translateY(32px)}
  to  {opacity:1;transform:translateY(0)}
}

/* verdict banner */
.verdict{
  border-radius:var(--r-sm);
  padding:20px 22px;margin-bottom:28px;
  display:flex;align-items:flex-start;gap:16px;
}
.verdict.safe{
  background:linear-gradient(135deg,rgba(16,185,129,0.10),rgba(52,211,153,0.06));
  border:1px solid rgba(52,211,153,0.30);
}
.verdict.warn{
  background:linear-gradient(135deg,rgba(244,63,94,0.10),rgba(251,146,60,0.06));
  border:1px solid rgba(244,63,94,0.30);
}
.verdict-icon{
  width:48px;height:48px;flex-shrink:0;
  border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:22px;
}
.verdict.safe .verdict-icon{background:rgba(52,211,153,0.12)}
.verdict.warn .verdict-icon{background:rgba(244,63,94,0.12)}
.verdict-body{}
.verdict-tag{
  font-family:var(--mono);font-size:10px;letter-spacing:1.5px;
  text-transform:uppercase;color:var(--text3);margin-bottom:4px;
}
.verdict-title{font-size:20px;font-weight:700;letter-spacing:-0.3px;line-height:1.2}
.verdict.safe .verdict-title{color:#34d399}
.verdict.warn .verdict-title{color:#fb7185}
.verdict-sub{font-size:13px;color:var(--text2);margin-top:5px;line-height:1.5}

/* ── Confidence ring ── */
.conf-row{
  display:flex;align-items:center;gap:24px;
  margin-bottom:28px;
  padding:20px;
  background:rgba(255,255,255,0.02);
  border:1px solid var(--border);border-radius:var(--r-sm);
}
.conf-ring{flex-shrink:0}
.ring-svg{width:90px;height:90px;transform:rotate(-90deg)}
.ring-track{fill:none;stroke:rgba(255,255,255,0.06);stroke-width:7}
.ring-fill{
  fill:none;stroke-width:7;stroke-linecap:round;
  stroke-dasharray:238;stroke-dashoffset:238;
  transition:stroke-dashoffset 1.2s cubic-bezier(0.16,1,0.3,1), stroke .4s;
}
.ring-text-wrap{
  position:relative;width:90px;height:90px;margin-top:-90px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  pointer-events:none;
}
.ring-pct{
  font-size:18px;font-weight:700;font-family:var(--mono);
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.ring-lbl{font-size:9px;color:var(--text3);font-family:var(--mono);letter-spacing:1px;text-transform:uppercase}
.conf-detail{flex:1}
.conf-heading{font-size:14px;font-weight:600;color:var(--text);margin-bottom:4px}
.conf-desc{font-size:12px;color:var(--text2);line-height:1.6}
.threshold-badge{
  display:inline-block;margin-top:8px;
  background:rgba(129,140,248,0.12);
  border:1px solid rgba(129,140,248,0.25);
  border-radius:6px;padding:3px 10px;
  font-family:var(--mono);font-size:11px;color:var(--accent);
}

/* ── Probability bars ── */
.prob-section-title{
  font-family:var(--mono);font-size:10px;letter-spacing:2px;
  color:var(--text3);text-transform:uppercase;margin-bottom:16px;
}
.prob-list{display:flex;flex-direction:column;gap:16px}
.prob-item{}
.prob-header{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:8px;
}
.prob-name{font-size:14px;font-weight:600}
.prob-badge{
  display:flex;align-items:center;gap:6px;
  font-family:var(--mono);font-size:12px;
}
.prob-val{font-weight:500}
.prob-top-tag{
  background:rgba(99,102,241,0.15);
  border:1px solid rgba(99,102,241,0.3);
  border-radius:4px;padding:2px 8px;
  font-size:10px;color:var(--accent);
  font-family:var(--mono);letter-spacing:1px;
}
.bar-track{
  height:10px;background:rgba(255,255,255,0.05);
  border-radius:999px;overflow:visible;position:relative;
}
.bar-fill{
  height:100%;border-radius:999px;
  width:0%; /* animated via JS */
  transition:width 1.0s cubic-bezier(0.16,1,0.3,1);
  position:relative;
}
.bar-fill::after{
  content:'';position:absolute;right:-1px;top:50%;transform:translateY(-50%);
  width:10px;height:10px;border-radius:50%;
  background:inherit;box-shadow:0 0 10px currentColor;
  opacity:0;transition:opacity .3s .8s;
}
.bar-fill.loaded::after{opacity:1}

/* ── Assessment report ── */
.assessment-report{
  margin-top:24px;
  padding:18px 20px;
  background:rgba(99,102,241,0.05);
  border:1px solid rgba(99,102,241,0.16);
  border-radius:var(--r-sm);
}
.report-head{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:10px;
}
.report-title{
  font-family:var(--mono);font-size:10px;letter-spacing:2px;
  color:var(--accent);text-transform:uppercase;
}
.report-certainty{
  font-family:var(--mono);font-size:10px;letter-spacing:1px;
  padding:3px 9px;border-radius:999px;
  border:1px solid currentColor;text-transform:uppercase;
}
.report-summary{
  font-size:13px;color:var(--text);line-height:1.7;margin-bottom:12px;
}
.report-factors{
  margin:0;padding-left:18px;
  font-size:12.5px;color:var(--text2);line-height:1.8;
}
.report-factors li{margin-bottom:6px}

/* ── Clinical note ── */
.clinical-note{
  margin-top:24px;
  padding:16px 20px;
  background:rgba(251,191,36,0.04);
  border:1px solid rgba(251,191,36,0.14);
  border-left:3px solid rgba(251,191,36,0.5);
  border-radius:0 var(--r-sm) var(--r-sm) 0;
  font-size:12px;color:var(--text2);line-height:1.7;
}
.clinical-note strong{color:#fbbf24}

/* ═══════════════════════════════════════════════════
   MODEL SPECS  (bottom card)
═══════════════════════════════════════════════════ */
.specs-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(155px,1fr));
  gap:12px;
}
.spec-item{
  background:rgba(255,255,255,0.02);
  border:1px solid var(--border);
  border-radius:var(--r-sm);
  padding:16px 18px;
  transition:border-color .2s,background .2s;
}
.spec-item:hover{
  border-color:rgba(129,140,248,0.3);
  background:rgba(99,102,241,0.05);
}
.spec-label{
  font-family:var(--mono);font-size:10px;
  letter-spacing:1.5px;text-transform:uppercase;
  color:var(--text3);margin-bottom:6px;
}
.spec-val{
  font-size:15px;font-weight:600;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}

/* ═══════════════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════════════ */
footer{
  text-align:center;padding:40px 0 0;
  font-family:var(--mono);font-size:11px;color:var(--text3);
  animation:fade-up 1s ease .5s both;
}
footer a{color:var(--text3);text-decoration:none;border-bottom:1px solid var(--text3)}

/* ═══════════════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════════════ */
@media(max-width:640px){
  .hero{padding:48px 0 36px}
  .card{padding:22px}
  .conf-row{flex-direction:column;align-items:flex-start;gap:16px}
  .specs-grid{grid-template-columns:repeat(2,1fr)}
}
@media(prefers-reduced-motion:reduce){
  *{animation-duration:.01ms!important;transition-duration:.01ms!important}
}
</style>
</head>
<body>

<!-- ░░░ AMBIENT ORBS ░░░ -->
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>

<!-- ░░░ NEURAL CANVAS ░░░ -->
<canvas id="neural-bg"></canvas>

<div class="page-wrap">

  <!-- ══════════ HERO ══════════ -->
  <header class="hero">
    <div class="hero-chip">NeuroScan · SSBD Detection</div>
    <h1>Autism Behavior<br><span class="grad">Analysis System</span></h1>
    <p class="hero-sub">
      Upload a video clip and our AI pipeline — MobileNetV2 feature extraction
      paired with an LSTM sequence classifier — instantly screens for
      self-stimulatory behaviors associated with autism spectrum disorder.
    </p>
    <div class="hero-meta">
      <div class="meta-pill">Model F1 <span>84.0 ± 3.7</span></div>
      <div class="meta-pill">Frames <span>90/clip</span></div>
      <div class="meta-pill">Confidence gate <span>55%</span></div>
      <div class="meta-pill">SSBD Dataset <span>3 classes</span></div>
    </div>
  </header>

  <!-- ══════════ UPLOAD CARD ══════════ -->
  <div class="card">
    <div class="section-label">Upload Video</div>

    <div class="drop-zone" id="drop-zone">
      <input type="file" id="video-input" accept="video/*" aria-label="Select video file">
      <div class="dz-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="url(#icon-grad)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
          <defs>
            <linearGradient id="icon-grad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#818cf8"/>
              <stop offset="100%" stop-color="#38bdf8"/>
            </linearGradient>
          </defs>
          <rect x="2" y="3" width="20" height="14" rx="2"/>
          <line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
          <polygon points="10,8 16,11 10,14"/>
        </svg>
      </div>
      <div class="dz-title">Drop your video here</div>
      <div class="dz-hint">or <strong>click to browse</strong> your files</div>
      <div class="dz-formats">MP4 · AVI · MOV · WEBM — Max 200 MB</div>
      <div id="file-name"></div>
    </div>

    <button class="btn-analyze" id="submit-btn" disabled aria-label="Analyze video">
      &#9650;&nbsp; Run Analysis
    </button>

    <!-- progress -->
    <div class="progress-wrap" id="progress-wrap">
      <div class="progress-steps" id="progress-steps">
        <div class="p-step" id="ps0">Upload</div>
        <div class="p-step" id="ps1">Frames</div>
        <div class="p-step" id="ps2">CNN</div>
        <div class="p-step" id="ps3">LSTM</div>
        <div class="p-step" id="ps4">Verdict</div>
      </div>
      <div class="progress-track">
        <div class="progress-fill" id="progress-fill"></div>
      </div>
      <div class="scan-label" id="scan-label">Initialising…</div>
    </div>

    <div class="error-box" id="error-box"></div>
  </div>

  <!-- ══════════ RESULT CARD ══════════ -->
  <div class="card slide-in" id="result-card">
    <div class="section-label">Detection Result</div>

    <!-- verdict -->
    <div class="verdict" id="verdict">
      <div class="verdict-icon" id="verdict-icon"></div>
      <div class="verdict-body">
        <div class="verdict-tag" id="verdict-tag"></div>
        <div class="verdict-title" id="verdict-title"></div>
        <div class="verdict-sub" id="verdict-sub"></div>
      </div>
    </div>

    <!-- confidence ring -->
    <div class="conf-row">
      <div class="conf-ring">
        <svg class="ring-svg" viewBox="0 0 80 80">
          <circle class="ring-track" cx="40" cy="40" r="31"/>
          <circle class="ring-fill" id="ring-fill" cx="40" cy="40" r="31"/>
        </svg>
        <div class="ring-text-wrap">
          <div class="ring-pct" id="ring-pct">0%</div>
          <div class="ring-lbl">Conf.</div>
        </div>
      </div>
      <div class="conf-detail">
        <div class="conf-heading">Model Confidence</div>
        <div class="conf-desc" id="conf-desc"></div>
        <div class="threshold-badge" id="thresh-badge"></div>
      </div>
    </div>

    <!-- probability bars -->
    <div class="prob-section-title">Class Probabilities</div>
    <div class="prob-list" id="prob-list"></div>

    <!-- assessment report -->
    <div class="assessment-report" id="assessment-report">
      <div class="report-head">
        <span class="report-title">Assessment Report</span>
        <span class="report-certainty" id="report-certainty"></span>
      </div>
      <div class="report-summary" id="report-summary"></div>
      <ul class="report-factors" id="report-factors"></ul>
    </div>

    <!-- clinical note -->
    <div class="clinical-note" id="clinical-note"></div>
  </div>

  <!-- ══════════ MODEL SPECS ══════════ -->
  <div class="card">
    <div class="section-label">Model Architecture</div>
    <div class="specs-grid">
      <div class="spec-item"><div class="spec-label">Extractor</div><div class="spec-val">MobileNetV2</div></div>
      <div class="spec-item"><div class="spec-label">Feature Dim</div><div class="spec-val">1,280-D</div></div>
      <div class="spec-item"><div class="spec-label">Frames/Clip</div><div class="spec-val">90</div></div>
      <div class="spec-item"><div class="spec-label">LSTM Hidden</div><div class="spec-val">64 Units</div></div>
      <div class="spec-item"><div class="spec-label">Dropout</div><div class="spec-val">30%</div></div>
      <div class="spec-item"><div class="spec-label">Optimizer</div><div class="spec-val">Adam 1e-2</div></div>
      <div class="spec-item"><div class="spec-label">Conf. Gate</div><div class="spec-val">55%</div></div>
      <div class="spec-item"><div class="spec-label">CV Folds</div><div class="spec-val">5-Fold</div></div>
      <div class="spec-item"><div class="spec-label">Seeds</div><div class="spec-val">100</div></div>
      <div class="spec-item"><div class="spec-label">Paper F1</div><div class="spec-val">84.0 ±3.7</div></div>
    </div>
  </div>

  <footer>
    Based on:
    <a href="#" target="_blank">Lakkapragada et al., JMIR Biomed Eng 2022;7(1):e33771</a>
    &nbsp;·&nbsp; Research Prototype — Not a clinical diagnostic tool
  </footer>
</div><!-- /page-wrap -->

<script>
/* ════════════════════════════════════════════════
   NEURAL NETWORK BACKGROUND  (canvas animation)
════════════════════════════════════════════════ */
(function(){
  const canvas = document.getElementById('neural-bg');
  const ctx    = canvas.getContext('2d');
  let W, H, nodes=[], RAF;

  const NODE_COUNT = 70;
  const MAX_DIST   = 160;
  const COLORS     = ['#6366f1','#38bdf8','#818cf8','#34d399'];

  function resize(){
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function rand(a,b){ return a + Math.random()*(b-a); }

  function init(){
    nodes=[];
    for(let i=0;i<NODE_COUNT;i++){
      nodes.push({
        x:rand(0,W), y:rand(0,H),
        vx:rand(-0.22,0.22), vy:rand(-0.22,0.22),
        r:rand(1.4,3.0),
        color:COLORS[Math.floor(Math.random()*COLORS.length)],
        alpha:rand(0.2,0.5)
      });
    }
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    // draw edges
    for(let i=0;i<nodes.length;i++){
      for(let j=i+1;j<nodes.length;j++){
        const dx=nodes[i].x-nodes[j].x, dy=nodes[i].y-nodes[j].y;
        const d=Math.sqrt(dx*dx+dy*dy);
        if(d<MAX_DIST){
          const a=(1-d/MAX_DIST)*0.18;
          ctx.beginPath();
          ctx.strokeStyle=`rgba(129,140,248,${a})`;
          ctx.lineWidth=0.7;
          ctx.moveTo(nodes[i].x,nodes[i].y);
          ctx.lineTo(nodes[j].x,nodes[j].y);
          ctx.stroke();
        }
      }
    }
    // draw nodes
    nodes.forEach(n=>{
      ctx.beginPath();
      ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
      ctx.fillStyle=n.color;
      ctx.globalAlpha=n.alpha;
      ctx.fill();
      ctx.globalAlpha=1;
    });
  }

  function step(){
    nodes.forEach(n=>{
      n.x+=n.vx; n.y+=n.vy;
      if(n.x<-10||n.x>W+10) n.vx*=-1;
      if(n.y<-10||n.y>H+10) n.vy*=-1;
    });
    draw();
    RAF=requestAnimationFrame(step);
  }

  window.addEventListener('resize',()=>{ resize(); init(); });
  resize(); init(); step();
})();

/* ════════════════════════════════════════════════
   SCROLL REVEAL
════════════════════════════════════════════════ */
const revealObserver = new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      e.target.style.opacity='1';
      e.target.style.transform='translateY(0)';
    }
  });
},{threshold:0.08});
document.querySelectorAll('.card').forEach(c=>{
  c.style.opacity='0';c.style.transform='translateY(28px)';
  c.style.transition='opacity .6s ease, transform .6s ease';
  revealObserver.observe(c);
});

/* ════════════════════════════════════════════════
   UPLOAD & DRAG/DROP
════════════════════════════════════════════════ */
const classes  = {{ classes|tojson }};
const clrMap   = {{ colors|tojson }};

const fileInput  = document.getElementById('video-input');
const fileName   = document.getElementById('file-name');
const submitBtn  = document.getElementById('submit-btn');
const dropZone   = document.getElementById('drop-zone');
const progressW  = document.getElementById('progress-wrap');
const progressF  = document.getElementById('progress-fill');
const scanLabel  = document.getElementById('scan-label');
const resultCard = document.getElementById('result-card');
const errorBox   = document.getElementById('error-box');

dropZone.addEventListener('dragover',  e=>{e.preventDefault();dropZone.classList.add('drag-over')});
dropZone.addEventListener('dragleave', ()=>dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e=>{
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if(e.dataTransfer.files.length){ fileInput.files=e.dataTransfer.files; handleFile(e.dataTransfer.files[0]); }
});
fileInput.addEventListener('change',()=>{ if(fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(f){
  fileName.innerHTML = `<div class="dz-selected">&#10003;&nbsp; ${f.name} &nbsp;(${(f.size/1024/1024).toFixed(1)} MB)</div>`;
  submitBtn.disabled = false;
  resultCard.style.display = 'none';
  errorBox.style.display   = 'none';
}

/* ════════════════════════════════════════════════
   PROGRESS STEPPER
════════════════════════════════════════════════ */
const stepData = [
  [18, 'Uploading video to server…',        0],
  [38, 'Sampling frames (90 per clip)…',    1],
  [62, 'MobileNetV2 → 1280-D features…',   2],
  [84, 'LSTM sequence classification…',     3],
  [96, 'Applying 55% confidence gate…',     4],
];
let stepTimer, curStep=0;

function startProgress(){
  curStep=0; progressF.style.width='0%';
  progressW.style.display='block';
  document.querySelectorAll('.p-step').forEach(s=>s.className='p-step');
  stepTimer = setInterval(()=>{
    if(curStep<stepData.length){
      const [pct,label,idx]=stepData[curStep++];
      progressF.style.width=pct+'%';
      scanLabel.textContent=label;
      document.querySelectorAll('.p-step').forEach((s,i)=>{
        if(i<idx) s.className='p-step done';
        else if(i===idx) s.className='p-step active';
        else s.className='p-step';
      });
    }
  },1300);
}

function stopProgress(){
  clearInterval(stepTimer);
  progressF.style.width='100%';
  scanLabel.textContent='Analysis complete.';
  document.querySelectorAll('.p-step').forEach(s=>s.className='p-step done');
  setTimeout(()=>{ progressW.style.display='none'; },800);
}

/* ════════════════════════════════════════════════
   SUBMIT
════════════════════════════════════════════════ */
submitBtn.addEventListener('click', async()=>{
  if(!fileInput.files.length) return;
  submitBtn.disabled=true;
  errorBox.style.display='none';
  resultCard.style.display='none';
  startProgress();

  const fd=new FormData();
  fd.append('video', fileInput.files[0]);

  try{
    const res  = await fetch('/predict',{method:'POST',body:fd});
    const data = await res.json();
    stopProgress();
    if(data.error){ showError(data.error); }
    else           { renderResult(data); }
  }catch(err){
    stopProgress();
    showError('Network error — is the Flask server running?');
  }finally{
    submitBtn.disabled=false;
  }
});

function showError(msg){
  errorBox.textContent='⚠  '+msg;
  errorBox.style.display='block';
}

/* ════════════════════════════════════════════════
   RENDER RESULT
════════════════════════════════════════════════ */
function renderResult(data){
  const confPct  = (data.max_confidence*100).toFixed(1);
  const confRaw  = data.max_confidence;
  const gate     = (data.confidence_threshold*100).toFixed(0);

  /* — verdict banner — */
  const verdict      = document.getElementById('verdict');
  const vIcon        = document.getElementById('verdict-icon');
  const vTag         = document.getElementById('verdict-tag');
  const vTitle       = document.getElementById('verdict-title');
  const vSub         = document.getElementById('verdict-sub');

  if(!data.autism_detected){
    verdict.className='verdict safe';
    vIcon.textContent='✅';
    vTag.textContent='Result · No Behavior';
    vTitle.textContent='No Autism-Related Behavior Detected';
    vSub.textContent=
      `Model confidence peaked at ${confPct}% — below the ${gate}% threshold. `+
      `The video does not strongly resemble arm flapping, head banging, or spinning.`;
  }else{
    const idx=data.label_idx;
    const behaviorColor=clrMap[idx];
    verdict.className='verdict warn';
    vIcon.textContent=['🤲','🤕','🌀'][idx];
    vTag.textContent='Result · Behavior Detected';
    vTitle.textContent='Behavior Detected: '+data.label;
    vTitle.style.color=behaviorColor;
    vSub.textContent=
      `Confidence ${confPct}% exceeds threshold ${gate}%. `+
      `${data.num_frames} frames processed. Predicted behavior: ${data.label}.`;
  }

  /* — confidence ring — */
  const ringFill = document.getElementById('ring-fill');
  const CIRC=194.78; // 2π×31
  const offset=CIRC-(confRaw*CIRC);
  const ringColor = data.autism_detected ? '#f43f5e' : '#34d399';
  ringFill.style.stroke=ringColor;
  ringFill.style.strokeDashoffset=CIRC; // start at 0
  setTimeout(()=>ringFill.style.strokeDashoffset=offset,80);
  document.getElementById('ring-pct').textContent=confPct+'%';

  const confDesc = document.getElementById('conf-desc');
  const threshBadge = document.getElementById('thresh-badge');
  confDesc.textContent = data.autism_detected
    ? `The classifier is ${confPct}% confident in the predicted behavior class.`
    : `All class probabilities are below the ${gate}% threshold — no behavior pattern matched.`;
  threshBadge.textContent=`Confidence threshold: ${gate}%`;

  /* — probability bars — */
  const probList = document.getElementById('prob-list');
  probList.innerHTML='';
  data.probabilities.forEach((p,i)=>{
    const pct=(p*100).toFixed(1);
    const isTop=(i===data.label_idx) && data.autism_detected;
    const color=clrMap[i];
    probList.innerHTML+=`
      <div class="prob-item">
        <div class="prob-header">
          <span class="prob-name" style="color:${color}">${classes[i]}</span>
          <div class="prob-badge">
            <span class="prob-val" style="color:${color}">${pct}%</span>
            ${isTop?'<span class="prob-top-tag">DETECTED</span>':''}
          </div>
        </div>
        <div class="bar-track">
          <div class="bar-fill" id="bar-${i}"
               style="background:linear-gradient(90deg,${color}99,${color});width:0%;opacity:${isTop?1:0.45}"></div>
        </div>
      </div>`;
  });
  // animate bars after DOM paint
  setTimeout(()=>{
    data.probabilities.forEach((p,i)=>{
      const bar=document.getElementById('bar-'+i);
      bar.style.width=(p*100).toFixed(1)+'%';
      setTimeout(()=>bar.classList.add('loaded'),900);
    });
  },120);

  /* — assessment report — */
  if(data.report){
    const certColor = {High:'#34d399', Moderate:'#fbbf24', Low:'#f87171'}[data.report.certainty] || '#94a3b8';
    const certBadge = document.getElementById('report-certainty');
    certBadge.textContent = data.report.certainty + ' certainty';
    certBadge.style.color = certColor;

    document.getElementById('report-summary').textContent = data.report.summary;

    const factorsEl = document.getElementById('report-factors');
    factorsEl.innerHTML = '';
    data.report.factors.forEach(f=>{
      factorsEl.innerHTML += `<li>${f}</li>`;
    });
  }

  /* — clinical note — */
  const note = document.getElementById('clinical-note');
  if(!data.autism_detected){
    note.innerHTML=`<strong>Note:</strong> This result indicates no self-stimulatory behavior was detected with sufficient confidence. 
    This is a research prototype built on the SSBD dataset — it is <strong>not a clinical diagnostic tool</strong>. 
    Always consult a licensed clinician for any autism-related assessment.`;
  }else{
    note.innerHTML=`<strong>Clinical Disclaimer:</strong> Detection of self-stimulatory behavior does not constitute a 
    diagnosis of Autism Spectrum Disorder. This system is a <strong>research prototype</strong> (JMIR Biomed Eng 2022;7(1):e33771) 
    for screening purposes only. Please refer findings to a qualified healthcare professional.`;
  }

  /* — show card with animation — */
  resultCard.style.display='block';
  resultCard.style.animation='none';
  void resultCard.offsetHeight; // reflow
  resultCard.style.animation='slide-in-up .55s cubic-bezier(0.16,1,0.3,1) both';
  setTimeout(()=>resultCard.scrollIntoView({behavior:'smooth',block:'start'}),80);
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, classes=CLASSES, colors=COLORS)

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    ext      = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_name = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}{ext}")
    file.save(tmp_name)
    try:
        result = detector.predict(tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    if result.get("error"):
        return jsonify(result), 422
    return jsonify(result)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": CLASSES})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
