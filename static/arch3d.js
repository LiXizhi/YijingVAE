// Shared Three.js visualization for the VAE tensor pipeline.
// Renders each stage as a 3D plate; consecutive plates are joined by their
// four corners, producing the hourglass / X silhouette of an encoder–decoder.
//
// Usage (in a <script type="module"> after declaring an importmap for "three"):
//
//   import { createArchViz } from '/static/arch3d.js';
//   const viz = createArchViz({
//     canvas: document.getElementById('arch3d'),
//     tip:    document.getElementById('archTip'),
//     empty:  document.getElementById('archEmpty'),
//   });
//   viz.update(archJson, batchSize);

import * as THREE from 'three';

const ROLE_COLOR = {
  io:  0xb8bfcd,
  enc: 0x5b86c4,
  lat: 0xd7c98a,
  dec: 0x5fa572,
};
const ROLE_NAME = { io: '输入/输出', enc: '编码器', lat: '潜变量', dec: '解码器' };

function defaultRoleOf(i, n) {
  if (i === 0 || i === n - 1) return 'io';
  if (i >= 6 && i <= 7) return 'lat';
  if (i < 6) return 'enc';
  return 'dec';
}

function plateSize(shape) {
  if (shape.length === 3) {
    const [c, h, w] = shape;
    const s = Math.log2(Math.max(2, Math.max(h, w))) * 0.9;
    return { w: s * (w / Math.max(h, w)), h: s * (h / Math.max(h, w)), c };
  }
  if (shape.length === 1) {
    const d = shape[0];
    const side = Math.max(0.4, Math.log2(Math.max(2, d)) * 0.45);
    return { w: side, h: side, c: 1 };
  }
  return { w: 1, h: 1, c: 1 };
}

export function createArchViz({ canvas, tip, empty, roleOf = defaultRoleOf }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
  camera.position.set(0, 6, 22);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 0.6);
  key.position.set(5, 10, 8); scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.35);
  rim.position.set(-6, -4, -8); scene.add(rim);

  const layerGroup = new THREE.Group();
  scene.add(layerGroup);

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let hoverTarget = null;

  let yaw = -0.55, pitch = 0.28, dragging = false, lastX = 0, lastY = 0;

  canvas.addEventListener('pointerdown', e => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', e => {
    dragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
  });
  canvas.addEventListener('pointermove', e => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    if (dragging) {
      yaw   += (e.clientX - lastX) * 0.008;
      pitch += (e.clientY - lastY) * 0.006;
      pitch = Math.max(-0.9, Math.min(0.9, pitch));
      lastX = e.clientX; lastY = e.clientY;
    }
  });
  canvas.addEventListener('pointerleave', () => {
    hoverTarget = null;
    if (tip) tip.style.opacity = 0;
  });

  function resize3D() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    }
  }
  window.addEventListener('resize', resize3D);

  let lastSig = '';
  function update(arch, batch) {
    const sig = JSON.stringify([arch && arch.shapes, batch]);
    if (sig === lastSig) return;
    lastSig = sig;

    while (layerGroup.children.length) {
      const o = layerGroup.children.pop();
      if (o.geometry) o.geometry.dispose();
      if (o.material && o.material.dispose) o.material.dispose();
    }

    if (!arch || !arch.shapes) {
      if (empty) empty.style.display = 'flex';
      return;
    }
    if (empty) empty.style.display = 'none';

    const shapes  = arch.shapes;
    const n       = shapes.length;
    const plates  = shapes.map(s => plateSize(s.shape));
    const spacing = 1.6;
    const x0      = -(n - 1) * spacing / 2;

    for (let i = 0; i < n; i++) {
      const role  = roleOf(i, n);
      const color = ROLE_COLOR[role];
      const p     = plates[i];
      const x     = x0 + i * spacing;
      const depth = Math.max(0.08, Math.min(1.2, Math.log2(Math.max(2, p.c)) * 0.16));

      const geom = new THREE.BoxGeometry(depth, Math.max(0.15, p.h), Math.max(0.15, p.w));
      const mat  = new THREE.MeshPhongMaterial({
        color, transparent: true, opacity: 0.78,
        shininess: 40, emissive: color, emissiveIntensity: 0.12,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(x, 0, 0);
      mesh.userData = {
        stage: shapes[i].stage,
        note:  shapes[i].note,
        shape: shapes[i].shape,
        role,
        batch,
      };
      layerGroup.add(mesh);

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geom),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }),
      );
      edges.position.copy(mesh.position);
      edges.userData = { ignore: true };
      layerGroup.add(edges);

      if (i < n - 1) {
        const nx = x0 + (i + 1) * spacing;
        const np = plates[i + 1];
        const a = Math.max(0.15, p.h)  / 2, b = Math.max(0.15, p.w)  / 2;
        const c = Math.max(0.15, np.h) / 2, d = Math.max(0.15, np.w) / 2;
        const corners = [
          [ a,  b,  c,  d], [ a, -b,  c, -d],
          [-a,  b, -c,  d], [-a, -b, -c, -d],
        ];
        const arr = [];
        corners.forEach(([y1, z1, y2, z2]) => {
          arr.push(x + depth / 2, y1, z1, nx - depth / 2, y2, z2);
        });
        const lineGeom = new THREE.BufferGeometry();
        lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
        const lines = new THREE.LineSegments(lineGeom,
          new THREE.LineBasicMaterial({ color: 0x3a4a78, transparent: true, opacity: 0.5 }));
        lines.userData = { ignore: true };
        layerGroup.add(lines);
      }
    }
  }

  function updateHover() {
    if (!tip) return;
    const meshes = layerGroup.children.filter(o => o.isMesh);
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(meshes, false);
    if (hits.length) {
      const m = hits[0].object;
      if (hoverTarget !== m) {
        if (hoverTarget) hoverTarget.material.opacity = 0.78;
        hoverTarget = m;
        m.material.opacity = 1.0;
      }
      const d = m.userData;
      const b = d.batch ? String(d.batch) : 'B';
      const shapeStr = '(' + [b, ...d.shape].join(' × ') + ')';
      tip.innerHTML = `<div class="t">${d.stage}</div>
                       <div class="s">${shapeStr}</div>
                       <div class="n">${d.note} · ${ROLE_NAME[d.role]}</div>`;
      const v = m.position.clone().applyMatrix4(layerGroup.matrixWorld).project(camera);
      const rect = canvas.getBoundingClientRect();
      const sx = (v.x * 0.5 + 0.5) * rect.width;
      const sy = (-v.y * 0.5 + 0.5) * rect.height;
      tip.style.left = sx + 'px';
      tip.style.top  = sy + 'px';
      tip.style.opacity = 1;
    } else {
      if (hoverTarget) hoverTarget.material.opacity = 0.78;
      hoverTarget = null;
      tip.style.opacity = 0;
    }
  }

  function tick() {
    resize3D();
    layerGroup.rotation.y = yaw;
    layerGroup.rotation.x = pitch;
    layerGroup.updateMatrixWorld();
    updateHover();
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  tick();

  return { update };
}
