// WebGL2 渲染器：按表面标签着色、碎片分显、点选、编号精灵
import type { Mesh, RigidTransform, Vec3 } from '../core/types';
import { SurfaceLabel } from '../core/types';
import { mat4LookAt, mat4Mul, mat4Perspective, rigidToMat4, v3 } from '../core/math3d';

const VERT = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in float aLabel;
uniform mat4 uModel;
uniform mat4 uViewProj;
uniform vec4 uColorOverride; // xyz 颜色，w<0 表示不覆盖
out vec3 vNormal;
out vec3 vWorld;
out float vLabel;
void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vWorld = world.xyz;
  vNormal = mat3(uModel) * aNormal;
  vLabel = aLabel;
  gl_Position = uViewProj * world;
}`;

const FRAG = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vWorld;
in float vLabel;
uniform vec4 uColorOverride;
uniform vec3 uEye;
uniform int uShardMode; // 0 按标签 1 单色 2 按碎片配色
uniform vec3 uShardColor;
uniform float uSelected;
out vec4 outColor;

vec3 labelColor(float l) {
  if (l < 0.5) return vec3(0.78, 0.45, 0.28); // 釉面 赭褐
  if (l < 1.5) return vec3(0.62, 0.52, 0.42); // 胎面
  if (l < 2.5) return vec3(0.85, 0.78, 0.66); // 断口 米白
  if (l < 3.5) return vec3(0.45, 0.40, 0.36); // 底足
  return vec3(0.36, 0.62, 0.85);              // 补配 青蓝
}

void main() {
  vec3 base = uColorOverride.w >= 0.0 ? uColorOverride.xyz
           : uShardMode == 1 ? uShardColor
           : uShardMode == 2 ? uShardColor
           : labelColor(vLabel);
  vec3 n = normalize(vNormal);
  vec3 lightDir = normalize(vec3(0.5, 0.8, 0.6));
  float diff = max(dot(n, lightDir), 0.0);
  float amb = 0.42;
  vec3 viewDir = normalize(uEye - vWorld);
  float spec = (vLabel > 3.5) ? 0.05 : pow(max(dot(reflect(-lightDir, n), viewDir), 0.0), 24.0) * 0.25;
  vec3 col = base * (amb + diff * 0.62) + vec3(spec);
  if (uSelected > 0.5) col = mix(col, vec3(1.0, 0.85, 0.25), 0.45);
  outColor = vec4(col, 1.0);
}`;

const LINE_VERT = `#version 300 es
layout(location=0) in vec3 aPos;
uniform mat4 uViewProj;
void main() { gl_Position = uViewProj * vec4(aPos, 1.0); }`;
const LINE_FRAG = `#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 outColor;
void main() { outColor = vec4(uColor, 1.0); }`;

const PICK_VERT = `#version 300 es
layout(location=0) in vec3 aPos;
uniform mat4 uModel;
uniform mat4 uViewProj;
void main() { gl_Position = uViewProj * uModel * vec4(aPos, 1.0); }`;
const PICK_FRAG = `#version 300 es
precision highp float;
uniform vec3 uId;
out vec4 outColor;
void main() { outColor = vec4(uId, 1.0); }`;

export interface RenderableShard {
  id: number;
  code: string;
  mesh: Mesh;
  transform: RigidTransform;
  color: Vec3;
  visible: boolean;
}

export interface CameraState {
  yaw: number;
  pitch: number;
  distance: number;
  target: Vec3;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('着色器编译失败: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function program(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('程序链接失败: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

function computeNormals(mesh: Mesh): Float32Array {
  const n = mesh.positions.length / 3;
  const normals = new Float32Array(n * 3);
  const p = mesh.positions;
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const ia = mesh.indices[i] * 3, ib = mesh.indices[i + 1] * 3, ic = mesh.indices[i + 2] * 3;
    const ux = p[ib] - p[ia], uy = p[ib + 1] - p[ia + 1], uz = p[ib + 2] - p[ia + 2];
    const vx = p[ic] - p[ia], vy = p[ic + 1] - p[ia + 1], vz = p[ic + 2] - p[ia + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
    normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
    normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
  }
  for (let i = 0; i < n; i++) {
    const x = normals[i * 3], y = normals[i * 3 + 1], z = normals[i * 3 + 2];
    const l = Math.hypot(x, y, z) || 1;
    normals[i * 3] = x / l; normals[i * 3 + 1] = y / l; normals[i * 3 + 2] = z / l;
  }
  return normals;
}

interface GpuMesh {
  vao: WebGLVertexArrayObject;
  count: number;
  normals: Float32Array;
  centroid: Vec3;
}

export class Renderer {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private lineProg: WebGLProgram;
  private pickProg: WebGLProgram;
  private gpu = new Map<number, GpuMesh>();
  private fillGpu: GpuMesh | null = null;
  private fillMesh: Mesh | null = null;
  camera: CameraState = { yaw: 0.6, pitch: 0.35, distance: 360, target: [0, 65, 0] };
  private width = 1;
  private height = 1;
  showFill = true;
  showFractures = true;
  shardMode: 0 | 1 | 2 = 0;
  selectedId = -1;
  private pickTarget = new Uint8Array(4);

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true });
    if (!gl) throw new Error('当前浏览器不支持 WebGL2');
    this.gl = gl;
    this.prog = program(gl, VERT, FRAG);
    this.lineProg = program(gl, LINE_VERT, LINE_FRAG);
    this.pickProg = program(gl, PICK_VERT, PICK_FRAG);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.clearColor(0.13, 0.14, 0.16, 1);
  }

  resize(): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.width = w;
    this.height = h;
    this.gl.viewport(0, 0, w, h);
  }

  private upload(mesh: Mesh): GpuMesh {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const normals = computeNormals(mesh);

    const posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const nBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nBuf);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    const lBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, lBuf);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.labels, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.UNSIGNED_BYTE, true, 0, 0);

    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);

    gl.bindVertexArray(null);
    let cx = 0, cy = 0, cz = 0;
    const n = mesh.positions.length / 3;
    for (let i = 0; i < n; i++) {
      cx += mesh.positions[i * 3]; cy += mesh.positions[i * 3 + 1]; cz += mesh.positions[i * 3 + 2];
    }
    return { vao, count: mesh.indices.length, normals, centroid: [cx / n, cy / n, cz / n] };
  }

  setShards(shards: RenderableShard[]): void {
    this.gpu.clear();
    for (const s of shards) {
      this.gpu.set(s.id, this.upload(s.mesh));
    }
  }

  setFillMesh(mesh: Mesh | null): void {
    this.fillMesh = mesh;
    this.fillGpu = mesh ? this.upload(mesh) : null;
  }

  private viewProj(): { vp: Float32Array; eye: Vec3 } {
    const c = this.camera;
    const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
    const eye: Vec3 = [
      c.target[0] + c.distance * cp * Math.sin(c.yaw),
      c.target[1] + c.distance * sp,
      c.target[2] + c.distance * cp * Math.cos(c.yaw),
    ];
    const view = mat4LookAt(eye, c.target, [0, 1, 0]);
    const proj = mat4Perspective(Math.PI / 4, this.width / Math.max(1, this.height), 1, 4000);
    return { vp: mat4Mul(proj, view), eye };
  }

  render(shards: RenderableShard[], linePoints: Float32Array | null): void {
    const gl = this.gl;
    this.resize();
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const { vp, eye } = this.viewProj();

    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, 'uViewProj'), false, vp);
    gl.uniform3fv(gl.getUniformLocation(this.prog, 'uEye'), eye);
    const locOverride = gl.getUniformLocation(this.prog, 'uColorOverride');
    const locMode = gl.getUniformLocation(this.prog, 'uShardMode');
    const locShardColor = gl.getUniformLocation(this.prog, 'uShardColor');
    const locSel = gl.getUniformLocation(this.prog, 'uSelected');
    const locModel = gl.getUniformLocation(this.prog, 'uModel');

    for (const s of shards) {
      if (!s.visible) continue;
      const g = this.gpu.get(s.id);
      if (!g) continue;
      gl.bindVertexArray(g.vao);
      gl.uniformMatrix4fv(locModel, false, rigidToMat4(s.transform));
      gl.uniform4f(locOverride, 0, 0, 0, -1);
      gl.uniform1i(locMode, this.shardMode);
      gl.uniform3fv(locShardColor, s.color);
      gl.uniform1f(locSel, s.id === this.selectedId ? 1 : 0);
      gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
    }

    if (this.showFill && this.fillGpu && this.fillMesh) {
      gl.bindVertexArray(this.fillGpu.vao);
      gl.uniformMatrix4fv(locModel, false, new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
      gl.uniform4f(locOverride, 0.36, 0.62, 0.85, 1);
      gl.uniform1i(locMode, 0);
      gl.uniform1f(locSel, 0);
      gl.drawElements(gl.TRIANGLES, this.fillGpu.count, gl.UNSIGNED_INT, 0);
    }

    if (linePoints && linePoints.length >= 6) {
      this.drawLines(linePoints, [0.95, 0.92, 0.55]);
    }
    gl.bindVertexArray(null);
  }

  private lineVao: WebGLVertexArrayObject | null = null;
  private lineBuf: WebGLBuffer | null = null;
  private lineCount = 0;

  private drawLines(points: Float32Array, color: Vec3): void {
    const gl = this.gl;
    if (!this.lineVao) {
      this.lineVao = gl.createVertexArray();
      this.lineBuf = gl.createBuffer();
    }
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, points, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.useProgram(this.lineProg);
    const { vp } = this.viewProj();
    gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, 'uViewProj'), false, vp);
    gl.uniform3fv(gl.getUniformLocation(this.lineProg, 'uColor'), color);
    gl.drawArrays(gl.LINE_LOOP, 0, points.length / 3);
    gl.bindVertexArray(null);
  }

  /** 屏幕坐标拾取碎片 id，未命中返回 -1 */
  pick(shards: RenderableShard[], sx: number, sy: number): number {
    const gl = this.gl;
    const dpr = this.canvas.width / Math.max(1, this.canvas.clientWidth);
    const x = sx * dpr;
    const y = this.canvas.height - sy * dpr;
    const { vp } = this.viewProj();
    gl.useProgram(this.pickProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.pickProg, 'uViewProj'), false, vp);
    const locModel = gl.getUniformLocation(this.pickProg, 'uModel');
    const locId = gl.getUniformLocation(this.pickProg, 'uId');
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(x, y, 1, 1);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.CULL_FACE);
    for (const s of shards) {
      if (!s.visible) continue;
      const g = this.gpu.get(s.id);
      if (!g) continue;
      gl.bindVertexArray(g.vao);
      gl.uniformMatrix4fv(locModel, false, rigidToMat4(s.transform));
      gl.uniform3f(locId, ((s.id + 1) & 255) / 255, (((s.id + 1) >> 8) & 255) / 255, (((s.id + 1) >> 16) & 255) / 255);
      gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
    }
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.pickTarget);
    gl.disable(gl.SCISSOR_TEST);
    gl.enable(gl.CULL_FACE);
    gl.clearColor(0.13, 0.14, 0.16, 1);
    const id = this.pickTarget[0] | (this.pickTarget[1] << 8) | (this.pickTarget[2] << 16);
    return id > 0 ? id - 1 : -1;
  }

  /** 世界坐标 -> 屏幕坐标（供编号标注定位） */
  worldToScreen(p: Vec3): { x: number; y: number; depth: number } {
    const { vp } = this.viewProj();
    const cx = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
    const cy = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
    const cw = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
    const ndcX = cx / cw, ndcY = cy / cw, ndcZ = (vp[2] * p[0] + vp[6] * p[1] + vp[10] * p[2] + vp[14]) / cw;
    const dpr = this.canvas.width / Math.max(1, this.canvas.clientWidth);
    return {
      x: (ndcX * 0.5 + 0.5) * this.canvas.width / dpr,
      y: (1 - (ndcY * 0.5 + 0.5)) * this.canvas.height / dpr,
      depth: ndcZ,
    };
  }

  getFillMesh(): Mesh | null {
    return this.fillMesh;
  }
}

/** 碎片伪随机但稳定的配色（碎片分显模式） */
export function shardColor(id: number): Vec3 {
  const h = (id * 2654435761) >>> 0;
  return [
    0.45 + ((h & 255) / 255) * 0.4,
    0.45 + (((h >> 8) & 255) / 255) * 0.4,
    0.45 + (((h >> 16) & 255) / 255) * 0.4,
  ];
}

export { SurfaceLabel, v3 };
