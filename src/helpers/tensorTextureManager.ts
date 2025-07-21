import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';
import { RootState } from '@react-three/fiber';

export class TensorTextureManager {
  private texture: THREE.DataTexture;
  private width: number;
  private height: number;
  private backend: tf.MathBackendWebGL;

  constructor(width = 1, height = 1) {
    this.width = width;
    this.height = height;
    this.backend = tf.backend() as tf.MathBackendWebGL;

    // Create a proper Three.js DataTexture
    const data = new Float32Array(width * height * 4);
    this.texture = new THREE.DataTexture(
      data,
      width,
      height,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    this.texture.needsUpdate = true;
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.minFilter = THREE.NearestFilter;
  }

  updateFromTensor(tensor: tf.Tensor): void {
    // Ensure tensor is on GPU
    this.backend.uploadToGPU(tensor.dataId);

    // Get the WebGL texture from TensorFlow.js
    const gpuData = this.backend.readToGPU(tensor.dataId);
    const tfTexture = gpuData.texture!;

    // Get WebGL context
    const gl = this.backend.gpgpu.gl;

    // Create framebuffer for texture copy
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      tfTexture,
      0
    );

    // Read pixels from TensorFlow texture
    const pixels = new Float32Array(this.width * this.height * 4);
    gl.readPixels(
      0, 0,
      this.width, this.height,
      gl.RGBA,
      gl.FLOAT,
      pixels
    );

    // Clean up
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(framebuffer);

    // Update Three.js texture data
    const textureData = this.texture.image.data as Float32Array;
    textureData.set(pixels);
    this.texture.needsUpdate = true;
  }

  // Alternative: Direct texture binding (more efficient but trickier)
  bindTensorTexture(state: RootState, tensor: tf.Tensor): void {
    // Ensure tensor is on GPU
    this.backend.uploadToGPU(tensor.dataId);

    // Get the WebGL texture from TensorFlow.js
    const gpuData = this.backend.readToGPU(tensor.dataId);
    const tfTexture = gpuData.texture;

    const gl = this.backend.gpgpu.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    
    // Get Three.js WebGL properties
    const properties = state.gl.properties;
    const textureProperties = properties.get(this.texture);

    // Store the current texture for restoration
    const previousTexture = textureProperties.__webglTexture;

    // Temporarily bind the TensorFlow texture
    textureProperties.__webglTexture = tfTexture;

    // Force texture update on next render
    //this.texture.version++;
  }

  getTexture(): THREE.Texture {
    return this.texture;
  }
}
