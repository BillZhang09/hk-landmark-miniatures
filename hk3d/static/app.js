import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
};

const state = { lat: null, lon: null, info: null, plan: null, job: null };

/* ------------------------------------------------------------------ map */
const LANDSD = "https://mapapi.geodata.gov.hk/gs/api/v1.0.0/xyz";
const ATTR = '<a href="https://api.portal.hkmapservice.gov.hk/disclaimer" target="_blank">地圖由地政總署提供 Map from Lands Department</a>';
const map = L.map("map", { zoomControl: true, minZoom: 10, maxZoom: 19 }).setView([22.3105, 114.2245], 15);
const basemap = L.tileLayer(`${LANDSD}/basemap/wgs84/{z}/{x}/{y}.png`, { attribution: ATTR, maxZoom: 20, maxNativeZoom: 20 });
const imagery = L.tileLayer(`${LANDSD}/imagery/wgs84/{z}/{x}/{y}.png`, { attribution: ATTR.replace("地圖", "航攝照片"), maxZoom: 20, maxNativeZoom: 19 });
const labels = L.tileLayer(`${LANDSD}/label/hk/tc/wgs84/{z}/{x}/{y}.png`, { maxZoom: 20, maxNativeZoom: 20 });
basemap.addTo(map);
labels.addTo(map);
new ResizeObserver(() => map.invalidateSize()).observe(document.getElementById("map"));
$("imagery").addEventListener("change", (e) => {
  if (e.target.checked) { map.removeLayer(basemap); imagery.addTo(map); }
  else { map.removeLayer(imagery); basemap.addTo(map); }
  labels.bringToFront();
});

const sheetLayer = L.geoJSON(null, {
  style: { color: "#4c7a4a", weight: 1, fillColor: "#4c7a4a", fillOpacity: 0.06, dashArray: "3 3" },
  onEachFeature: (f, l) => l.bindTooltip(`${f.properties.name}（本地）`, { sticky: true }),
}).addTo(map);
const needLayer = L.geoJSON(null, {
  style: (f) => f.properties.local ? { color: "#4c7a4a", weight: 1.5, fillOpacity: 0 } :
    { color: "#b5452c", weight: 1.5, fillColor: "#b5452c", fillOpacity: 0.12 },
  onEachFeature: (f, l) => l.bindTooltip(`${f.properties.name}${f.properties.local ? "" : "（缺少）"}`, { sticky: true }),
}).addTo(map);

let circle = null, marker = null;
function setCenter(lat, lon, pan = false) {
  state.lat = lat; state.lon = lon;
  const r = +$("radius").value;
  if (!circle) {
    circle = L.circle([lat, lon], { radius: r, color: "#b5452c", weight: 2, fillOpacity: 0.08 }).addTo(map);
    marker = L.marker([lat, lon], { draggable: true }).addTo(map);
    marker.on("drag", (e) => circle.setLatLng(e.latlng));
    marker.on("dragend", (e) => setCenter(e.target.getLatLng().lat, e.target.getLatLng().lng));
  } else {
    circle.setLatLng([lat, lon]); marker.setLatLng([lat, lon]);
  }
  circle.setRadius(r);
  if (pan) map.fitBounds(circle.getBounds(), { padding: [40, 40] });
  $("c-wgs").textContent = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
  refreshPlan();
}
map.on("click", async (e) => {
  const { lat, lng } = e.latlng;
  if ($("snap").checked) {
    try {
      const r = await api(`/api/snap?lat=${lat}&lon=${lng}`);
      if (r.building) return setCenter(r.building.lat, r.building.lon);
    } catch { /* fall through to the raw click */ }
  }
  setCenter(lat, lng);
});

const DIRS = ["北", "東北", "東", "東南", "南", "西南", "西", "西北"];
function direction(dx, dy) {
  return DIRS[Math.round(((Math.atan2(dx, dy) * 180 / Math.PI) + 360) % 360 / 45) % 8];
}
let nearbyTimer = null;
function refreshNearby() {
  clearTimeout(nearbyTimer);
  nearbyTimer = setTimeout(async () => {
    if (state.lat == null) return;
    const r = Math.max(600, 3 * +$("radius").value);
    let items = [];
    try { items = await api(`/api/landmarks?lat=${state.lat}&lon=${state.lon}&radius_m=${r}&limit=8`); } catch { }
    const ul = $("nearby-list");
    ul.replaceChildren(...items.map((b) => {
      const li = document.createElement("li");
      const here = b.dist_m < 5;
      li.className = here ? "here" : "";
      const p = state.plan;
      const where = here ? "目前中心" : p ? `${direction(b.e - p.e, b.n - p.n)} ${Math.round(b.dist_m)} m` : `${Math.round(b.dist_m)} m`;
      li.innerHTML = `${Math.round(b.height_m)} m<small>${where}</small>`;
      li.title = `頂部 ${b.top_m} mPD · 平面約 ${Math.round(b.width_m)}×${Math.round(b.depth_m)} m · 圖幅 ${b.sheet}`;
      li.addEventListener("click", () => setCenter(b.lat, b.lon, true));
      return li;
    }));
    $("nearby").hidden = !items.length;
  }, 300);
}

let planTimer = null;
function refreshPlan() {
  clearTimeout(planTimer);
  planTimer = setTimeout(async () => {
    if (state.lat == null) return;
    const r = +$("radius").value;
    try {
      const p = await api(`/api/plan?lat=${state.lat}&lon=${state.lon}&radius_m=${r}`);
      state.plan = p;
      $("c-hk").textContent = `E ${p.e.toFixed(1)}  N ${p.n.toFixed(1)}`;
      $("c-sheets").textContent = p.sheets.join("、") || "—";
      refreshNearby();
      needLayer.clearLayers().addData(p.geojson);
      const basic = p.basic || [];
      $("missing").hidden = !p.missing.length && !basic.length;
      $("missing-p").hidden = !p.missing.length;
      $("missing-list").textContent = p.missing.join("、");
      $("basic-p").hidden = !basic.length;
      $("basic-list").textContent = basic.join("、");
      if (!p.missing.length && basic.length) $("dl-format").value = "FBX";
      updateDlButton();
      $("go").disabled = p.coverage === 0 || state.busy;
      $("go").title = p.coverage === 0 ? "此範圍沒有本地數據" : "";
    } catch (e) { showErr(e.message); }
  }, 150);
}

/* -------------------------------------------------------------- search */
let searchTimer = null, results = [];
$("q").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  if (q.length < 2) { $("results").hidden = true; return; }
  searchTimer = setTimeout(async () => {
    try {
      results = await api(`/api/search?q=${encodeURIComponent(q)}`);
      const ul = $("results");
      ul.innerHTML = "";
      results.forEach((r, i) => {
        const li = document.createElement("li");
        li.innerHTML = `${escapeHtml(r.label)}<small>${escapeHtml(r.sub)}</small>`;
        li.onclick = () => pick(i);
        ul.appendChild(li);
      });
      ul.hidden = !results.length;
    } catch (err) { showErr(`搜尋失敗：${err.message}`); }
  }, 350);
});
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter" && results.length) pick(0); });
document.addEventListener("click", (e) => { if (!e.target.closest(".search")) $("results").hidden = true; });
function pick(i) {
  const r = results[i];
  $("results").hidden = true;
  $("q").value = r.label;
  showTab("map");
  setCenter(r.lat, r.lon, true);
}
const escapeHtml = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------------------------------------------------------- controls */
function updateScaleHint() {
  const r = +$("radius").value, d = +$("diameter").value;
  $("o-radius").textContent = r >= 1000 ? `${(r / 1000).toFixed(2)} km` : `${r} m`;
  const scale = Math.round((2 * r * 1000) / d);
  const cell = Math.max(+$("detail_mm").value * (2 * r) / d, 0.1);
  $("scale-hint").textContent = `比例約 1:${scale.toLocaleString()} · 1 mm 打印 = ${(scale / 1000).toFixed(1)} m 實地 · 取樣 ${cell.toFixed(2)} m`;
  updateTileHint();
  if (circle) circle.setRadius(r);
}
// pieces across the bed, as the pipeline counts them (5 mm kept clear all round)
function updateTileHint() {
  const d = +$("diameter").value, shape = $("shape").value;
  const w = d, h = shape === "hexagon" ? d * Math.sqrt(3) / 2 : d;   // hexagon: corners east and west
  const nx = Math.max(1, Math.ceil(w / (+$("bed_x").value - 10) - 1e-9));
  const ny = Math.max(1, Math.ceil(h / (+$("bed_y").value - 10) - 1e-9));
  const big = nx * ny > 1;
  $("tile-hint").hidden = !big;
  if (big) $("tile-hint").textContent = $("tile").checked
    ? `打印直徑 ${d} mm 大過打印範圍：會切成 ${nx} × ${ny} 塊（太高的建築另外橫切），定位銷孔需要底座至少約 4.5 mm 厚。`
    : `打印直徑 ${d} mm 大過打印範圍，但未勾選「分塊打印」：輸出的單件模型放不進打印機。`;
}
["bed_x", "bed_y", "tile", "shape"].forEach((id) => $(id).addEventListener("change", updateTileHint));
["radius", "diameter", "detail_mm"].forEach((id) => $(id).addEventListener("input", () => { updateScaleHint(); if (id === "radius") refreshPlan(); }));

function renderLayers(groups) {
  const box = $("layers");
  box.innerHTML = "";
  for (const g of groups) {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" value="${g.key}" ${g.default ? "checked" : ""}><span class="sw" style="background:${g.color}"></span>${g.label}`;
    box.appendChild(l);
  }
}

function params() {
  const num = (id) => +$(id).value;
  const floor = $("floor_m").value.trim();
  return {
    lat: state.lat, lon: state.lon,
    radius_m: num("radius"), diameter_mm: num("diameter"), base_mm: num("base_mm"),
    z_scale: num("z_scale"), detail_mm: num("detail_mm"), min_feature_mm: num("min_feature_mm"),
    shape: $("shape").value, multicolor: $("multicolor").checked, skin_mm: num("skin_mm"),
    floor_m: floor === "" ? null : +floor, name: $("name").value.trim(), engrave: $("engrave").checked,
    ground_clean_m: num("ground_clean_m"), spire_mm: num("spire_mm"), line_mm: num("line_mm"), layer_mm: num("layer_mm"), openings_mm: num("openings_mm"), support_free: $("support_free").checked, bridge_mm: num("bridge_mm"), trim_edge: $("trim_edge").checked, precise_landmark: $("precise_landmark").checked, precise_all: $("precise_landmark").checked && $("precise_all").checked, facade_relief_mm: num("facade_relief_mm"), road_recess_layers: num("road_recess_layers"), holder: $("holder").checked,
    tile: $("tile").checked, bed_mm: [num("bed_x"), num("bed_y"), num("bed_z")], pin_mm: num("pin_mm"),
    groups: [...document.querySelectorAll("#layers input:checked")].map((i) => i.value),
  };
}

/* ---------------------------------------------------------------- jobs */
function showErr(msg) { $("err").hidden = !msg; $("err").textContent = msg || ""; }
function setBusy(b) {
  state.busy = b;
  $("go").disabled = b || !state.plan || state.plan.coverage === 0;
  $("dl-btn").disabled = b;
  $("prog").hidden = !b;
}
function setProgress(p, msg) {
  $("prog").querySelector("i").style.width = `${Math.round(p * 100)}%`;
  $("prog").querySelector("span").textContent = `${Math.round(p * 100)}% · ${msg}`;
}
async function waitJob(job) {
  while (job.state === "queued" || job.state === "running") {
    setProgress(job.progress, job.message);
    await new Promise((r) => setTimeout(r, 600));
    job = await api(`/api/jobs/${job.id}`);
  }
  if (job.state === "error") throw new Error(job.error);
  return job;
}

$("go").addEventListener("click", async () => {
  if (state.lat == null) return;
  showErr("");
  setBusy(true);
  try {
    const job = await waitJob(await api("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params()),
    }));
    showResult(job.key, job.result);
    loadHistory();
  } catch (e) { showErr(`生成失敗：${e.message}`); }
  finally { setBusy(false); }
});

function dlSheets() {
  const p = state.plan || {};
  const fmt = $("dl-format").value;
  return [...(p.missing || []), ...(fmt === "FBX" ? p.basic || [] : [])];
}
function updateDlButton() {
  const p = state.plan || {};
  const upgrade = $("dl-format").value === "FBX" && (p.basic || []).length;
  $("dl-btn").textContent = !(p.missing || []).length && upgrade ? "從地政總署下載有紋理版本（升級）"
    : upgrade ? "下載缺少的圖幅並升級無紋理圖幅" : "從地政總署下載缺少的圖幅";
  $("dl-btn").disabled = state.busy || !dlSheets().length;
}
$("dl-format").addEventListener("change", updateDlButton);

$("dl-btn").addEventListener("click", async () => {
  const miss = dlSheets();
  if (!miss.length) return;
  const fmt = $("dl-format").value;
  const what = fmt === "FBX" ? "有紋理 FBX（每幅約 1–2 GB）" : "無紋理 FBX0（每幅約 60 MB）";
  if (!confirm(`將從地政總署 Open3Dhk 下載 ${miss.length} 幅${what}，伺服器較慢，每幅可能需時數分鐘：\n${miss.join("、")}\n\n數據 © 香港特別行政區政府地政總署。繼續？`)) return;
  showErr("");
  setBusy(true);
  try {
    for (let i = 0; i < miss.length; i += 6) {
      await waitJob(await api("/api/downloads", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sheets: miss.slice(i, i + 6), format: fmt }),
      }));
    }
    await loadSheets();
    refreshPlan();
  } catch (e) { showErr(`下載失敗：${e.message}`); }
  finally { setBusy(false); }
});

function fmtBytes(n) { return n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${(n / 1e3).toFixed(0)} KB`; }

async function showResult(key, res) {
  const s = res.stats;
  $("out").hidden = false;
  const rows = [
    ["比例", s.scale], ["高度", `${s.height_mm} mm`], ["高程", `${s.elev_min_m} – ${s.elev_max_m} mPD`],
    ["取樣", `${s.cell_m} m（${s.grid_cells}²）`], ["三角面", s.triangles.toLocaleString()],
    ["體積", `${s.volume_cm3} cm³`], ["數據覆蓋", `${(s.data_coverage * 100).toFixed(1)}%`], ["耗時", `${s.seconds} s`],
  ];
  if (s.parts.length > 1) rows.push(["分件", s.parts.map((p) => p.name).join("、")]);
  const lm = s.landmark || {}, br = s.bridges || {};
  if (lm.support_mm3) rows.push(["免支撐補料", `${lm.support_mm3} mm³`]);
  if (br.bridge_open_m2) rows.push(["橋底開空", `${br.bridge_open_m2} m²`]);
  const tl = s.tiles || {};
  if (tl.pieces) rows.push(["分塊", `${tl.grid.replace("x", " × ")}，共 ${tl.pieces} 件（最多 ${tl.levels} 層）` +
    (tl.pins ? `，定位銷 ${tl.pins} 支 ⌀${tl.pin_mm} × ${tl.pin_length_mm} mm` : "") +
    (tl.base_raised_mm ? `；底座加厚至 ${tl.base_raised_mm} mm` : "") +
    (tl.floating ? `；${tl.floating} 處懸空部分需手動加支撐` : "")]);
  $("stats").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join("");
  const labels = { holder: "展示底座環 STL", stl: "STL 單件模型", "3mf": res.params.multicolor ? "3MF 多色分件" : "3MF 模型", parts_zip: "各分件 STL（ZIP）", pieces_zip: "分塊打印 STL＋拼裝圖（ZIP）", assembly: "拼裝圖 PNG", png: "俯視圖 PNG" };
  $("files").innerHTML = "";
  for (const [k, name] of Object.entries(res.files)) {
    if (!labels[k]) continue;
    const a = document.createElement("a");
    a.href = `/files/${key}/${encodeURIComponent(name)}`;
    a.download = name;
    a.innerHTML = `<span>${labels[k]}</span><small>${escapeHtml(name)}</small>`;
    $("files").appendChild(a);
  }
  const st = res.status || { level: "generated", issues: [] };
  const issueText = {
    landmark_failed: "中心地標精確雕刻失敗，改以普通方塊輸出",
    landmark_missing: "中心沒有可精確雕刻的建築",
    buildings_failed: "部分精確雕刻的建築失敗，改以普通方塊輸出",
    relief_failed: "外牆浮雕生成失敗，外牆為平面",
    relief_no_texture: "沒有外牆貼圖，外牆為平面",
    missing_data: "部分範圍沒有地圖數據，以周圍地面延伸填補",
    fragments_removed: "移除了懸空的碎片",
    floating_pieces: "分塊後有懸空部分，需要手動加支撐",
  };
  $("stats").insertAdjacentHTML("afterbegin",
    `<dt>狀態</dt><dd>${st.level === "generated" ? "已生成（未經切片或試印驗證）" : "已生成，需要檢查"}</dd>`);
  const msgs = st.issues.filter((i) => i.code !== "missing_data").map((i) => `${issueText[i.code] || i.code}（${i.detail}）`);
  if (s.data_coverage < 0.99) msgs.push(`範圍內只有 ${(s.data_coverage * 100).toFixed(1)}% 有數據，缺失部分（未下載的圖幅）以周圍地面延伸填補，未必準確；請先下載缺少的圖幅。`);
  if (msgs.length) showErr(`注意：${msgs.join("；")}`);
  showTab("view");
  await loadPreview(`/files/${key}/${res.files.glb}?t=${Date.now()}`);
}

/* ------------------------------------------------------------- viewer */
let renderer, scene, camera, controls, modelRoot;
function initViewer() {
  const el = $("viewer");
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  el.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(35, 1, 1, 5000);
  camera.up.set(0, 0, 1);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6f6658, 1.1));
  const sun = new THREE.DirectionalLight(0xfff6e8, 1.9);
  sun.position.set(-120, 160, 220);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  Object.assign(sun.shadow.camera, { left: -150, right: 150, top: 150, bottom: -150, near: 1, far: 800 });
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(2000, 2000), new THREE.ShadowMaterial({ opacity: 0.18 }));
  ground.position.z = -0.01;
  ground.receiveShadow = true;
  scene.add(ground);
  new ResizeObserver(resize).observe(el);
  (function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
}
function resize() {
  const el = $("viewer");
  const w = el.clientWidth, h = el.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
async function loadPreview(url) {
  if (!renderer) initViewer();
  $("view-empty").textContent = "載入預覽…";
  $("view-empty").hidden = false;
  const gltf = await new GLTFLoader().loadAsync(url);
  if (modelRoot) scene.remove(modelRoot);
  modelRoot = gltf.scene;
  // files are z-up millimetres; the camera uses z-up too, so no rotation needed
  const legend = [];
  modelRoot.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = o.receiveShadow = true;
      o.material.flatShading = false;
      o.geometry.computeVertexNormals();
      o.material.side = THREE.DoubleSide;
      const c = "#" + o.material.color.getHexString();
      if (!legend.some((l) => l.name === o.material.name)) legend.push({ name: o.material.name, c });
    }
  });
  scene.add(modelRoot);
  modelRoot.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(modelRoot);
  const ctr = box.getCenter(new THREE.Vector3());
  const r = box.getBoundingSphere(new THREE.Sphere()).radius;
  // fit the whole bounding sphere (tall towers included) from a 3/4 view
  const dist = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
  const dir = new THREE.Vector3(-0.45, -1, 0.7).normalize();
  controls.target.copy(ctr);
  camera.position.copy(ctr).addScaledVector(dir, dist);
  camera.near = r / 100; camera.far = r * 50; camera.updateProjectionMatrix();
  const names = { holder: "底座環", model: "模型", base: "底座／地形", building: "建築", infrastructure: "道路", water: "水體", vegetation: "植被" };
  $("legend").innerHTML = legend.map((l) => `<span><i style="background:${l.c}"></i>${names[l.name] || l.name}</span>`).join("");
  $("view-empty").hidden = true;
  resize();
}

/* ---------------------------------------------------------------- tabs */
function showTab(t) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("on", p.id === t));
  if (t === "map") map.invalidateSize();
  if (t === "view" && renderer) resize();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

/* ------------------------------------------------------------- presets */
const PRESETS = {
  // a single tower on a small disc, like a desk ornament
  landmark: { radius_m: 100, diameter_mm: 80, base_mm: 5, z_scale: 1, detail_mm: 0.05,
              shape: "circle", holder: true, engrave: true, ground_clean_m: 2.5, trim_edge: true, precise_landmark: true, precise_all: true, facade_relief_mm: 0.3,
              groups: ["terrain", "ground", "infrastructure", "bridge", "water", "vegetation", "building"] },
  city: { radius_m: 300, diameter_mm: 100, base_mm: 3, z_scale: 1, detail_mm: 0.1,
          shape: "circle", holder: false, engrave: true, ground_clean_m: 2.5, trim_edge: false, precise_landmark: false, precise_all: false, facade_relief_mm: 0,
          groups: ["terrain", "ground", "infrastructure", "bridge", "water", "building"] },
};
// thin-feature settings follow the nozzle: spires 1.5x, ground noise 0.75x,
// line width = nozzle, layer height = half of it (all editable afterwards)
function nozzleParams() {
  const n = +$("nozzle").value;
  return { spire_mm: +(1.5 * n).toFixed(2), min_feature_mm: +(0.75 * n).toFixed(2), line_mm: n, layer_mm: +(n / 2).toFixed(3) };
}
// the printer is the user's, not the preset's: presets keep these as they are
function printerParams() {
  const v = (id) => +$(id).value;
  return { spire_mm: v("spire_mm"), min_feature_mm: v("min_feature_mm"), line_mm: v("line_mm"),
           layer_mm: v("layer_mm"), support_free: $("support_free").checked, bridge_mm: v("bridge_mm"),
           tile: $("tile").checked, bed_mm: [v("bed_x"), v("bed_y"), v("bed_z")], pin_mm: v("pin_mm") };
}
const PRINTER_IDS = ["nozzle", "layer_mm", "line_mm", "bridge_mm", "support_free", "bed_x", "bed_y", "bed_z", "pin_mm", "tile"];
function savePrinter() {
  try {
    localStorage.setItem("hk3d.printer", JSON.stringify(Object.fromEntries(
      PRINTER_IDS.map((id) => [id, $(id).type === "checkbox" ? $(id).checked : $(id).value]))));
  } catch {}
}
$("nozzle").addEventListener("change", () => {
  const p = nozzleParams();
  $("spire_mm").value = p.spire_mm; $("min_feature_mm").value = p.min_feature_mm;
  $("line_mm").value = p.line_mm; $("layer_mm").value = p.layer_mm;
  savePrinter();
});
PRINTER_IDS.slice(1).forEach((id) => $(id).addEventListener("change", savePrinter));
try {
  const saved = JSON.parse(localStorage.getItem("hk3d.printer") || "null");
  const old = localStorage.getItem("hk3d.nozzle");
  if (saved) {
    $("nozzle").value = saved.nozzle; $("nozzle").dispatchEvent(new Event("change"));
    for (const id of PRINTER_IDS.slice(1)) {
      if (saved[id] == null) continue;
      if ($(id).type === "checkbox") $(id).checked = !!saved[id]; else $(id).value = saved[id];
    }
  } else if (old) { $("nozzle").value = old; $("nozzle").dispatchEvent(new Event("change")); }
} catch {}

document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
  const p = PRESETS[b.dataset.preset];
  applyParams({ ...p, ...printerParams(), name: $("name").value, multicolor: $("multicolor").checked,
                skin_mm: +$("skin_mm").value, floor_m: null });
  document.querySelectorAll("[data-preset]").forEach((x) => x.classList.toggle("on", x === b));
  $("snap").checked = b.dataset.preset === "landmark";
  if (circle) circle.setRadius(p.radius_m);
  refreshPlan();
}));

/* ------------------------------------------------------------- history */
function applyParams(p) {
  const set = (id, v) => { if (v != null && $(id)) $(id).value = v; };
  set("radius", p.radius_m); set("diameter", p.diameter_mm); set("base_mm", p.base_mm);
  set("z_scale", p.z_scale); set("detail_mm", String(p.detail_mm)); set("min_feature_mm", p.min_feature_mm);
  set("shape", p.shape); set("skin_mm", p.skin_mm); set("name", p.name);
  $("floor_m").value = p.floor_m ?? "";
  $("multicolor").checked = !!p.multicolor;
  $("engrave").checked = p.engrave !== false;
  $("holder").checked = !!p.holder;
  $("trim_edge").checked = !!p.trim_edge;
  $("precise_landmark").checked = !!p.precise_landmark;
  $("precise_all").checked = !!p.precise_all;
  if (p.ground_clean_m != null) $("ground_clean_m").value = p.ground_clean_m;
  $("spire_mm").value = p.spire_mm ?? 0;
  $("line_mm").value = p.line_mm ?? 0.4; $("layer_mm").value = p.layer_mm ?? 0.2;
  $("openings_mm").value = p.openings_mm ?? 1;
  $("road_recess_layers").value = p.road_recess_layers ?? 0;   // models from before had true heights
  $("support_free").checked = p.support_free ?? true;
  $("bridge_mm").value = p.bridge_mm ?? 10;
  $("tile").checked = p.tile ?? true;
  const bed = p.bed_mm || [256, 256, 256];
  $("bed_x").value = bed[0]; $("bed_y").value = bed[1]; $("bed_z").value = bed[2];
  $("pin_mm").value = p.pin_mm ?? 1.75;
  $("facade_relief_mm").value = p.facade_relief_mm ?? 0;
  document.querySelectorAll("#layers input").forEach((i) => { i.checked = p.groups.includes(i.value); });
  updateScaleHint();
}
async function loadHistory() {
  const items = await api("/api/history");
  $("history-sec").hidden = !items.length;
  const ul = $("history");
  ul.innerHTML = "";
  for (const h of items.slice(0, 8)) {
    const li = document.createElement("li");
    const p = h.params;
    const title = p.name || `${h.lat.toFixed(4)}, ${h.lon.toFixed(4)}`;
    li.innerHTML = `<img src="/files/${h.key}/${h.files.png}" alt="" loading="lazy"><div><b>${escapeHtml(title)}</b>` +
      `<small>R ${p.radius_m} m · ⌀${p.diameter_mm} mm · ${h.stats.scale}${p.multicolor ? " · 多色" : ""}</small></div>`;
    li.onclick = () => {
      applyParams(p);
      setCenter(h.lat, h.lon, true);
      showResult(h.key, h);
    };
    ul.appendChild(li);
  }
  return items;
}

/* ---------------------------------------------------------------- boot */
async function loadSheets() {
  const fc = await api("/api/sheets");
  sheetLayer.clearLayers().addData(fc);
  $("status").textContent = `本地數據：${fc.features.length} 幅 1:1000 圖幅`;
  return fc;
}
(async function boot() {
  try {
    state.info = await api("/api/info");
    renderLayers(state.info.groups);
    updateScaleHint();
    $("status").textContent = "正在索引本地數據…";
    const fc = await loadSheets();
    map.invalidateSize();
    const recent = await loadHistory();
    const qs = new URLSearchParams(location.search);
    if (qs.get("preset") && PRESETS[qs.get("preset")]) document.querySelector(`[data-preset=${qs.get("preset")}]`).click();
    if (qs.get("lat") && qs.get("lon")) {
      // shareable link: ?lat=22.2776&lon=114.1658&preset=landmark
      setCenter(+qs.get("lat"), +qs.get("lon"), true);
    } else if (recent.length) {
      // resume where the last model was made
      const h = recent[0];
      applyParams(h.params);
      setCenter(h.lat, h.lon, true);
    } else if (fc.features.length) {
      // centre of one real sheet (the bbox of several sheets may fall in the sea)
      const c = L.geoJSON(fc.features[0]).getBounds().getCenter();
      setCenter(c.lat, c.lng, true);
    }
  } catch (e) { $("status").textContent = `錯誤：${e.message}`; }
})();
