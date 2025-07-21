import * as THREE from "three";
import * as tf from "@tensorflow/tfjs";
import { RootState } from "@react-three/fiber";

const forceTextureInitialization = (function () {
  const material = new THREE.MeshBasicMaterial();
  const geometry = new THREE.BoxGeometry();
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(geometry, material));
  const camera = new THREE.Camera();

  return function forceTextureInitialization(renderer, texture) {
    material.map = texture;
    renderer.render(scene, camera);
  };
})();

/*
Converts pixels of a canvas's texture to a float32 TensorflowJS tensor

Args
-------
canvas : HTMLCanvasElement
  the input canvas to convert
*/
export function canvasToTensor(canvas: HTMLCanvasElement) {
  // Set WebGL backend
  const backend = tf.backend() as tf.MathBackendWebGL;
  const gl = backend.gpgpu.gl;

  // Initialize ThreeJS canvas texture object
  const canvasTexture = new THREE.CanvasTexture(canvas);
  const canvasTextureSource = canvasTexture.image;

  // Copy ThreeJS canvas texture to new WebGL texture
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    0x8814,
    gl.RGBA,
    gl.FLOAT,
    canvasTextureSource,
  );

  // Set texture parameters (important for NPOT textures)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  // Initialize the tensor
  const width = canvas.width;
  const height = canvas.height;
  const logicalShape = [width, height, 4];
  const tensor = tf.tensor(
    { texture, height, width, channels: "RGBA" },
    logicalShape,
  );
  return tensor;
}

/*
Converts a TensorflowJS tensor to a ThreeJS texture

Args
-------
tensor : tf.Tensor
  the input tensor to convert to a THREE.js texture
*/
export function tensorToTexture(state: RootState, tensor: tf.Tensor, currentTexture: THREE.Texture, uploadToGPU?: boolean): THREE.Texture | null {
  const backend = tf.backend() as tf.MathBackendWebGL; // Check backend
  if (uploadToGPU) {
    // In most cases, uploading to GPU should be left false
    backend.uploadToGPU(tensor.dataId);
  }
  
  const gl = backend.gpgpu.gl; // WebGL context object
  const renderer = state.gl; // WebGL renderer object
  // Get underlying texture backing the tensor
  const tensorData = backend.readToGPU(tensor.dataId);
  const tensorDataTexture = tensorData.texture;
  // Clean up existing frame buffer (I don't why this works but will address later)
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  
  const currentTextureProps = renderer.properties.get(currentTexture);
  //@ts-ignore
  currentTextureProps.__webglTexture = tensorDataTexture; // Access and update the Three JS texture's texture data
  return currentTexture
}



export function createTestCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 1;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgba(100, 50, 255, 0.5)";
  ctx.fillRect(0, 0, 1, 1);
  return canvas;
  
}

/*
Simple implementation of canvasToTensor
*/
export function createTestTensor(canvas?: HTMLCanvasElement) {
  if (!canvas) {
    const canvas = createTestCanvas();
  }

  const textureTensor = canvasToTensor(canvas);
  textureTensor.mul(255).print(); // [100, 50, 255, 128]
  return textureTensor;
}

export function createTestCanvasTexture(canvas?: HTMLCanvasElement) {
  if (!canvas) {
    const canvas = createTestCanvas();
  }

  const texture = new THREE.CanvasTexture(canvas, ...[,,,,,,], THREE.FloatType);
  return texture;
}

/*
Simple implementation of tensorToTexture
*/
export function createTextureFromTensor(canvas?: HTMLCanvasElement) {
  const tensor = createTestTensor(canvas);
  const texture = tensorToTexture(tensor);
  console.log(texture);
  return texture;
}
