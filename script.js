const canvas = document.getElementById('glCanvas');
const gl = canvas.getContext('webgl2', { antialias: true }) || canvas.getContext('webgl', { antialias: true }) || canvas.getContext('experimental-webgl', { antialias: true });

const speedEl = document.getElementById('speed');
const detailEl = document.getElementById('detail');
const glowEl = document.getElementById('glow');
const qualityEl = document.getElementById('quality');
const volumeFormEl = document.getElementById('volumeForm');
const zoomMinEl = document.getElementById('zoomMin');
const zoomMaxEl = document.getElementById('zoomMax');

const startBtn = document.getElementById('startBtn');
const benchBtn = document.getElementById('benchBtn');
const backBtn = document.getElementById('backBtn');
const fullViewBtn = document.getElementById('fullViewBtn');

const fpsEl = document.getElementById('fps');
const frameMsEl = document.getElementById('frameMs');
const avgFpsEl = document.getElementById('avgFps');
const lowFpsEl = document.getElementById('lowFps');
const scoreBarEl = document.getElementById('scoreBar');
const benchStatusEl = document.getElementById('benchStatus');

const gpuVendorEl = document.getElementById('gpuVendor');
const gpuRendererEl = document.getElementById('gpuRenderer');
const gpuWebglEl = document.getElementById('gpuWebgl');

if (!gl) {
  benchStatusEl.textContent = 'WebGL unavailable in this browser. Try enabling hardware acceleration or a different browser.';

  startBtn.addEventListener('click', () => {
    benchStatusEl.textContent = 'WebGL unavailable: cannot render on this device/browser.';
  });

  benchBtn.addEventListener('click', () => {
    benchStatusEl.textContent = 'WebGL unavailable: benchmark cannot run.';
  });

  fullViewBtn.addEventListener('click', () => {
    document.body.classList.add('immersive');
  });

  backBtn.addEventListener('click', () => {
    document.body.classList.remove('immersive');
  });

  window.__WEBGL_DISABLED__ = true;
}

if (window.__WEBGL_DISABLED__) {
  // Stop setup but keep UI controls available.
} else {

const dbg = gl.getExtension('WEBGL_debug_renderer_info');
if (dbg) {
  gpuVendorEl.textContent = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) || 'Unknown';
  gpuRendererEl.textContent = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || 'Unknown';
} else {
  gpuVendorEl.textContent = gl.getParameter(gl.VENDOR) || 'Unknown';
  gpuRendererEl.textContent = gl.getParameter(gl.RENDERER) || 'Unknown';
}
gpuWebglEl.textContent = `WebGL ${gl.getParameter(gl.VERSION)}`;

const vertexSrc = `
attribute vec2 aPos;
varying vec2 vUv;
void main(){
  vUv = aPos;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

// Kernel and search logic adapted to match the supplied volumeshader approach.
const fragmentSrc = `
precision highp float;
varying vec2 vUv;

uniform vec2 uRes;
uniform float uTime;
uniform float uSpeed;
uniform float uGlow;
uniform float uDetail;
uniform float uYaw;
uniform float uPitch;
uniform float uLen;
uniform float uForm;

mat2 rot(float a){
  float c = cos(a), s = sin(a);
  return mat2(c,-s,s,c);
}

vec3 hsv2rgb(vec3 c){
  vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float kernalClassic(vec3 ver){
  float x, y, z;
  x = cos(1.0 / (ver.x * ver.x + 0.06));
  y = cos(1.0 / (ver.y * ver.y + 0.06));
  z = cos(1.0 / (ver.z * ver.z + 0.06));
  return -x - y - z - 1.2;
}

float kernalRipple(vec3 ver){
  float r = length(ver);
  return sin(8.0 * r) + 0.35 * sin(12.0 * ver.x) + 0.35 * sin(12.0 * ver.y) + 0.35 * sin(12.0 * ver.z) - 0.6;
}

float kernalTorus(vec3 ver){
  float qx = length(ver.xz) - 0.9;
  float tor = length(vec2(qx, ver.y)) - 0.35;
  float twist = 0.15 * sin(7.0 * atan(ver.z, ver.x) + 3.5 * ver.y);
  return -(tor + twist);
}

float kernalLava(vec3 ver){
  float r = length(ver);
  float waves = sin(5.0 * ver.x + 2.1 * ver.y) + sin(5.2 * ver.y + 2.4 * ver.z) + sin(5.4 * ver.z + 2.7 * ver.x);
  return 1.15 - r + 0.18 * waves;
}

float kernalGyroid(vec3 ver){
  float g = sin(ver.x * 2.8) * cos(ver.y * 2.8) + sin(ver.y * 2.8) * cos(ver.z * 2.8) + sin(ver.z * 2.8) * cos(ver.x * 2.8);
  return g - 0.15;
}

float kernalBubble(vec3 ver){
  vec3 q = ver;
  for (int i = 0; i < 4; i++) {
    q = abs(q) / clamp(dot(q, q), 0.18, 10.0) - 0.72;
  }
  return 0.35 - length(q);
}

float kernal(vec3 ver){
  if (uForm < 0.5) return kernalClassic(ver);
  if (uForm < 1.5) return kernalRipple(ver);
  if (uForm < 2.5) return kernalTorus(ver);
  if (uForm < 3.5) return kernalLava(ver);
  if (uForm < 4.5) return kernalGyroid(ver);
  return kernalBubble(ver);
}

vec3 calcNormal(vec3 p, vec3 right, vec3 up, vec3 forward, float eps){
  float nx = kernal(p - right * eps) - kernal(p + right * eps);
  float ny = kernal(p - up * eps) - kernal(p + up * eps);
  float nz = kernal(p + forward * eps) - kernal(p - forward * eps);
  return normalize(vec3(nx, ny, nz));
}

bool solveSurface(vec3 ro, vec3 rd, float maxDist, out float hitT){
  // Fixed-step sign crossing search + binary refine (close to provided reference behavior)
  float stepLen = 0.01 * maxDist;
  float prevT = 0.0;
  float prevV = kernal(ro);

  // hard cap to keep predictable GPU load, detail control maps to max loop count
  float maxK = min(240.0, 22.0 + uDetail * 1.55);

  for (int k = 1; k < 260; k++) {
    if (float(k) > maxK) break;

    float t = stepLen * float(k);
    vec3 p = ro + rd * t;
    float v = kernal(p);

    if (v > 0.0 && prevV < 0.0) {
      float a = prevT;
      float b = t;
      for (int i = 0; i < 9; i++) {
        float m = 0.5 * (a + b);
        float mv = kernal(ro + rd * m);
        if (mv > 0.0) b = m;
        else a = m;
      }
      hitT = 0.5 * (a + b);
      return hitT < (2.0 * maxDist);
    }

    prevT = t;
    prevV = v;
  }

  hitT = -1.0;
  return false;
}

void main(){
  vec2 uv = vec2(vUv.x * (uRes.x / uRes.y), vUv.y);

  float len = uLen;
  float ang1 = uYaw + uTime * 0.08 * uSpeed;
  float ang2 = uPitch + sin(uTime * 0.12 * uSpeed) * 0.04;

  vec3 origin = vec3(
    len * cos(ang1) * cos(ang2),
    len * sin(ang2),
    len * sin(ang1) * cos(ang2)
  );

  vec3 right = normalize(vec3(sin(ang1), 0.0, -cos(ang1)));
  vec3 up = normalize(vec3(-sin(ang2) * cos(ang1), cos(ang2), -sin(ang2) * sin(ang1)));
  vec3 forward = normalize(vec3(-cos(ang1) * cos(ang2), -sin(ang2), -sin(ang1) * cos(ang2)));

  vec3 rd = normalize(forward + right * uv.x + up * uv.y);

  float t;
  vec3 col = vec3(0.0);

  if (solveSurface(origin, rd, len, t)) {
    vec3 pos = origin + rd * t;
    vec3 n = calcNormal(pos, right, up, forward, max(0.0009, 0.001 * t));

    vec3 lightDir = normalize(vec3(0.276, 0.920, 0.276));
    float diff = max(dot(n, lightDir), 0.0);

    vec3 view = normalize(-rd);
    vec3 refl = reflect(-view, n);
    float spec = pow(max(dot(refl, lightDir), 0.0), 32.0);

    float hue = fract(0.58 + 0.45 * n.x + 0.25 * n.y + 0.15 * sin(12.0 * pos.z));
    vec3 rainbow = hsv2rgb(vec3(hue, 0.90, 1.0));

    float fres = pow(1.0 - max(dot(view, n), 0.0), 2.2);
    col = rainbow * (0.35 + 0.95 * diff) + spec * 0.35 + fres * vec3(0.08, 0.75, 0.9);

    float glow = exp(-3.0 * t) * uGlow;
    col += rainbow * glow * 0.22;
  }

  col = pow(max(col, vec3(0.0)), vec3(0.9));
  gl_FragColor = vec4(col, 1.0);
}
`;

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) || 'shader compile failed');
  }
  return sh;
}

const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, vertexSrc));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragmentSrc));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
  throw new Error(gl.getProgramInfoLog(prog) || 'program link failed');
}

gl.useProgram(prog);

const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
  -1, -1,
  1, -1,
  -1, 1,
  -1, 1,
  1, -1,
  1, 1,
]), gl.STATIC_DRAW);

const aPos = gl.getAttribLocation(prog, 'aPos');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

const uRes = gl.getUniformLocation(prog, 'uRes');
const uTime = gl.getUniformLocation(prog, 'uTime');
const uSpeed = gl.getUniformLocation(prog, 'uSpeed');
const uGlow = gl.getUniformLocation(prog, 'uGlow');
const uDetail = gl.getUniformLocation(prog, 'uDetail');
const uYaw = gl.getUniformLocation(prog, 'uYaw');
const uPitch = gl.getUniformLocation(prog, 'uPitch');
const uLen = gl.getUniformLocation(prog, 'uLen');
const uForm = gl.getUniformLocation(prog, 'uForm');

function resize() {
  const q = parseFloat(qualityEl.value);
  const dpr = Math.min(window.devicePixelRatio || 1, 2) * q;
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener('resize', resize);
qualityEl.addEventListener('change', resize);
volumeFormEl.addEventListener('change', () => {
  benchStatusEl.textContent = `Volume form changed to ${volumeFormEl.options[volumeFormEl.selectedIndex].text}.`;
});

zoomMinEl.addEventListener('input', () => {
  if (parseFloat(zoomMinEl.value) >= parseFloat(zoomMaxEl.value)) {
    zoomMaxEl.value = (parseFloat(zoomMinEl.value) + 0.1).toFixed(1);
  }
  camLen = Math.max(parseFloat(zoomMinEl.value), camLen);
});

zoomMaxEl.addEventListener('input', () => {
  if (parseFloat(zoomMaxEl.value) <= parseFloat(zoomMinEl.value)) {
    zoomMinEl.value = (parseFloat(zoomMaxEl.value) - 0.1).toFixed(1);
  }
  camLen = Math.min(parseFloat(zoomMaxEl.value), camLen);
});

let running = false;
let last = performance.now();
let benchOn = false;
let benchStart = 0;
let samples = [];

let yaw = 2.8;
let pitch = 0.42;
let camLen = 2.5;
let dragging = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener('mousedown', (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
});
window.addEventListener('mouseup', () => {
  dragging = false;
});
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  yaw += dx * 0.005;
  pitch += dy * 0.004;
  pitch = Math.max(-1.2, Math.min(1.2, pitch));
  lastX = e.clientX;
  lastY = e.clientY;
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const minZoom = Math.min(parseFloat(zoomMinEl.value), parseFloat(zoomMaxEl.value) - 0.1);
  const maxZoom = Math.max(parseFloat(zoomMaxEl.value), minZoom + 0.1);
  camLen *= Math.exp(e.deltaY * 0.001);
  camLen = Math.max(minZoom, Math.min(maxZoom, camLen));
}, { passive: false });

function percentile(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const idx = Math.floor((p / 100) * (s.length - 1));
  return s[idx];
}

function loop(now) {
  const dt = now - last;
  last = now;

  const fps = 1000 / Math.max(dt, 0.0001);
  fpsEl.textContent = fps.toFixed(1);
  frameMsEl.textContent = `${dt.toFixed(2)}ms`;
  scoreBarEl.style.width = `${Math.min(100, (fps / 144) * 100).toFixed(0)}%`;

  if (benchOn) {
    samples.push(fps);
    const elapsed = (now - benchStart) / 1000;
    benchStatusEl.textContent = `Running benchmark... ${elapsed.toFixed(1)} / 10.0s`;

    if (elapsed >= 10) {
      benchOn = false;
      benchBtn.disabled = false;
      const avg = samples.reduce((a, b) => a + b, 0) / samples.length;
      const low = percentile(samples, 1);
      avgFpsEl.textContent = avg.toFixed(1);
      lowFpsEl.textContent = low.toFixed(1);
      benchStatusEl.textContent = `Done. ${samples.length} samples captured.`;
      samples = [];
    }
  }

  gl.uniform2f(uRes, canvas.width, canvas.height);
  gl.uniform1f(uTime, now * 0.001);
  gl.uniform1f(uSpeed, parseFloat(speedEl.value));
  gl.uniform1f(uGlow, parseFloat(glowEl.value));
  gl.uniform1f(uDetail, parseFloat(detailEl.value));
  gl.uniform1f(uYaw, yaw);
  gl.uniform1f(uPitch, pitch);
  gl.uniform1f(uLen, camLen);
  gl.uniform1f(uForm, parseFloat(volumeFormEl.value));

  gl.drawArrays(gl.TRIANGLES, 0, 6);

  if (running) requestAnimationFrame(loop);
}

startBtn.addEventListener('click', () => {
  if (running) {
    benchStatusEl.textContent = 'Volume is already rendering in dashboard mode.';
    return;
  }
  running = true;
  benchStatusEl.textContent = 'Rendering volume in dashboard mode...';
  last = performance.now();
  requestAnimationFrame(loop);
});

benchBtn.addEventListener('click', () => {
  if (!running) {
    running = true;
    last = performance.now();
    requestAnimationFrame(loop);
  }
  benchOn = true;
  benchBtn.disabled = true;
  samples = [];
  benchStart = performance.now();
  benchStatusEl.textContent = 'Benchmark starting...';
});


canvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();
  running = false;
  benchOn = false;
  benchBtn.disabled = false;
  benchStatusEl.textContent = 'WebGL context lost. Attempting recovery...';
});

canvas.addEventListener('webglcontextrestored', () => {
  benchStatusEl.textContent = 'WebGL context restored. Reload page to fully reinitialize renderer.';
});

fullViewBtn.addEventListener('click', () => {
  document.body.classList.add('immersive');
  resize();
});

backBtn.addEventListener('click', () => {
  document.body.classList.remove('immersive');
  resize();
});

resize();
gl.clearColor(0, 0, 0, 1);
gl.clear(gl.COLOR_BUFFER_BIT);

// Do not auto-render on load. User must press Start.
running = false;
benchStatusEl.textContent = 'Idle — press Start GPU Test to render.';

}
