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

function plateSize(shape, entry) {
  if (shape.length === 3) {
    const [c, h, w] = shape;
    const s = Math.log2(Math.max(2, Math.max(h, w))) * 0.9;
    return { w: s * (w / Math.max(h, w)), h: s * (h / Math.max(h, w)), c };
  }
  if (shape.length === 1) {
    const d = shape[0];
    // Flatten is just a tensor reshape, not a learned layer — draw it as a
    // small marker plate so it doesn't dominate the silhouette.
    const note  = (entry && entry.note)  || '';
    const stage = (entry && entry.stage) || '';
    const isFlatten = /flatten|展平|reshape|unflatten/i.test(note + ' ' + stage);
    if (isFlatten) {
      return { w: 0.5, h: 0.5, c: 1, flatten: true };
    }
    const side = Math.max(0.4, Math.log2(Math.max(2, d)) * 0.45);
    return { w: side, h: side, c: 1 };
  }
  return { w: 1, h: 1, c: 1 };
}

export function createArchViz({ canvas, tip, empty, roleOf = defaultRoleOf, dimOf = null }) {
  const isDim = (i, n, role) => !!(dimOf && dimOf(i, n, role));
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

  // Output-plate texture (the rightmost plate shows the current generated
  // image, so users can see what the decoder produced). Input-plate texture
  // mirrors that on the leftmost plate so the viewer can see the source
  // image flowing into the encoder.
  let outputPlate = null;       // the rightmost mesh in layerGroup
  let outputDecal = null;       // a thin Plane child glued to its +x face
  let outputTexture = null;     // CanvasTexture backed by `outTexCanvas`
  const outTexCanvas = document.createElement('canvas');
  outTexCanvas.width = outTexCanvas.height = 128;
  const outTexCtx = outTexCanvas.getContext('2d');
  let pendingOutSrc = null;     // image set before the output plate exists

  let inputPlate   = null;      // the leftmost mesh in layerGroup
  let inputDecal   = null;      // a thin Plane child glued to its -x face
  let inputTexture = null;
  const inTexCanvas = document.createElement('canvas');
  inTexCanvas.width = inTexCanvas.height = 128;
  const inTexCtx = inTexCanvas.getContext('2d');
  let pendingInSrc = null;

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
    outputPlate = null;
    outputDecal = null;
    inputPlate  = null;
    inputDecal  = null;
    // Note: textures / canvases are reused across rebuilds.

    if (!arch || !arch.shapes) {
      if (empty) empty.style.display = 'flex';
      return;
    }
    if (empty) empty.style.display = 'none';

    // Flatten / Unflatten are pure tensor reshapes (not learned layers); hide
    // them so the silhouette stays symmetric.
    const isReshapeStage = s => /flatten|展平|reshape|unflatten/i.test(
      ((s && s.note) || '') + ' ' + ((s && s.stage) || ''));
    const shapes  = arch.shapes.filter(s => !isReshapeStage(s));
    const n       = shapes.length;
    const plates  = shapes.map(s => plateSize(s.shape, s));
    const spacing = 1.6;
    const x0      = -(n - 1) * spacing / 2;

    for (let i = 0; i < n; i++) {
      const role  = roleOf(i, n);
      const color = ROLE_COLOR[role];
      const p     = plates[i];
      const x     = x0 + i * spacing;
      const depth = p.flatten
        ? 0.08
        : Math.max(0.08, Math.min(1.2, Math.log2(Math.max(2, p.c)) * 0.16));

      const dim = isDim(i, n, role);
      const baseOpacity = dim ? 0.15 : 0.78;
      const geom = new THREE.BoxGeometry(depth, Math.max(0.15, p.h), Math.max(0.15, p.w));
      const mat  = new THREE.MeshPhongMaterial({
        color, transparent: true, opacity: baseOpacity,
        shininess: 40, emissive: color,
        emissiveIntensity: dim ? 0.04 : 0.12,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(x, 0, 0);
      mesh.userData = {
        stage: shapes[i].stage,
        note:  shapes[i].note,
        shape: shapes[i].shape,
        role,
        batch,
        baseOpacity,
        dim,
      };
      layerGroup.add(mesh);

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geom),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: dim ? 0.2 : 0.9 }),
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
        const nextRole = roleOf(i + 1, n);
        const linkDim  = isDim(i, n, role) || isDim(i + 1, n, nextRole);
        const lines = new THREE.LineSegments(lineGeom,
          new THREE.LineBasicMaterial({ color: 0x3a4a78, transparent: true, opacity: linkDim ? 0.12 : 0.5 }));
        lines.userData = { ignore: true };
        layerGroup.add(lines);
      }

      // Remember the rightmost plate so we can paint the generated image on it.
      if (i === n - 1) {
        outputPlate = mesh;
        const decalGeom = new THREE.PlaneGeometry(
          Math.max(0.15, p.w), Math.max(0.15, p.h));
        const decalMat = new THREE.MeshBasicMaterial({
          color: 0xffffff, transparent: true, opacity: 0,
          side: THREE.DoubleSide,
        });
        outputDecal = new THREE.Mesh(decalGeom, decalMat);
        // Plane defaults to facing +z; rotate to face +x (front of the plate).
        outputDecal.rotation.y = Math.PI / 2;
        outputDecal.position.set(depth / 2 + 0.01, 0, 0);
        outputDecal.userData = { ignore: true };
        mesh.add(outputDecal);
        if (outputTexture) {
          decalMat.map = outputTexture;
          decalMat.opacity = 1;
          decalMat.needsUpdate = true;
        } else if (pendingOutSrc) {
          setOutputImage(pendingOutSrc);
        }
      }

      // Remember the leftmost plate so we can paint the source image on it.
      if (i === 0) {
        inputPlate = mesh;
        const decalGeom = new THREE.PlaneGeometry(
          Math.max(0.15, p.w), Math.max(0.15, p.h));
        const decalMat = new THREE.MeshBasicMaterial({
          color: 0xffffff, transparent: true, opacity: 0,
          side: THREE.DoubleSide,
        });
        inputDecal = new THREE.Mesh(decalGeom, decalMat);
        // Face -x (front of the leftmost plate, away from the encoder).
        inputDecal.rotation.y = -Math.PI / 2;
        inputDecal.position.set(-depth / 2 - 0.01, 0, 0);
        inputDecal.userData = { ignore: true };
        mesh.add(inputDecal);
        if (inputTexture) {
          decalMat.map = inputTexture;
          decalMat.opacity = 1;
          decalMat.needsUpdate = true;
        } else if (pendingInSrc) {
          setInputImage(pendingInSrc);
        }
      }
    }
  }

  function setOutputImage(src) {
    if (!src) return;
    pendingOutSrc = src;
    const im = new Image();
    im.crossOrigin = 'anonymous';
    im.onload = () => {
      const cw = outTexCanvas.width, ch = outTexCanvas.height;
      outTexCtx.fillStyle = '#10131a';
      outTexCtx.fillRect(0, 0, cw, ch);
      // Preserve the source aspect ratio inside the plate face.
      const r = Math.min(cw / im.width, ch / im.height);
      const dw = im.width  * r, dh = im.height * r;
      outTexCtx.imageSmoothingEnabled = false;
      outTexCtx.drawImage(im, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
      if (!outputTexture) {
        outputTexture = new THREE.CanvasTexture(outTexCanvas);
        outputTexture.colorSpace = THREE.SRGBColorSpace;
        outputTexture.magFilter = THREE.NearestFilter;
      } else {
        outputTexture.needsUpdate = true;
      }
      if (outputDecal) {
        outputDecal.material.map = outputTexture;
        outputDecal.material.opacity = 1;
        outputDecal.material.needsUpdate = true;
      }
    };
    im.src = src;
  }

  function setInputImage(src) {
    if (!src) return;
    pendingInSrc = src;
    const im = new Image();
    im.crossOrigin = 'anonymous';
    im.onload = () => {
      const cw = inTexCanvas.width, ch = inTexCanvas.height;
      inTexCtx.fillStyle = '#10131a';
      inTexCtx.fillRect(0, 0, cw, ch);
      const r = Math.min(cw / im.width, ch / im.height);
      const dw = im.width  * r, dh = im.height * r;
      inTexCtx.imageSmoothingEnabled = false;
      inTexCtx.drawImage(im, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
      if (!inputTexture) {
        inputTexture = new THREE.CanvasTexture(inTexCanvas);
        inputTexture.colorSpace = THREE.SRGBColorSpace;
        inputTexture.magFilter = THREE.NearestFilter;
      } else {
        inputTexture.needsUpdate = true;
      }
      if (inputDecal) {
        inputDecal.material.map = inputTexture;
        inputDecal.material.opacity = 1;
        inputDecal.material.needsUpdate = true;
      }
    };
    im.src = src;
  }

  function updateHover() {
    if (!tip) return;
    const meshes = layerGroup.children.filter(o => o.isMesh);
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(meshes, false);
    if (hits.length) {
      const m = hits[0].object;
      if (hoverTarget !== m) {
        if (hoverTarget) hoverTarget.material.opacity = hoverTarget.userData.baseOpacity ?? 0.78;
        hoverTarget = m;
        m.material.opacity = m.userData.dim ? 0.5 : 1.0;
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
      if (hoverTarget) hoverTarget.material.opacity = hoverTarget.userData.baseOpacity ?? 0.78;
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

  return { update, setOutputImage, setInputImage };
}
