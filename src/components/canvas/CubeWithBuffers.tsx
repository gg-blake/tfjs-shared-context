"use client";

import { useRef, useEffect, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import * as tf from "@tensorflow/tfjs";
import { useSharedWebGLContext } from "@/helpers/hooks/useSharedWebGLContext";

// GPU-only texture to buffer conversion using transform feedback
function textureToBuffer(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  targetBuffer: WebGLBuffer,
  options: {
    width: number;
    height: number;
    components: number;
    vertexCount: number;
  },
): boolean {
  try {
    // Create shaders for transform feedback
    const vertexShaderSource = `#version 300 es
      in vec2 a_position;
      uniform sampler2D u_texture;
      uniform vec2 u_textureSize;
      
      out vec3 v_output;
      
      void main() {
        // Calculate texture coordinate from vertex index
        float index = float(gl_VertexID);
        vec2 texCoord = vec2(
          mod(index, u_textureSize.x) / u_textureSize.x,
          floor(index / u_textureSize.x) / u_textureSize.y
        );
        
        // Sample texture data
        vec4 texelData = texture(u_texture, texCoord);
        v_output = texelData.xyz; // Extract RGB as position data
        
        gl_Position = vec4(0.0); // Not used for transform feedback
      }
    `;

    const fragmentShaderSource = `#version 300 es
      precision mediump float;
      void main() {
        discard; // Not rendering to screen
      }
    `;

    // Create and compile shaders
    const vertexShader = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);

    // Create program with transform feedback
    const program = gl.createProgram()!;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);

    // Specify transform feedback varyings BEFORE linking
    gl.transformFeedbackVaryings(program, ["v_output"], gl.INTERLEAVED_ATTRIBS);
    gl.linkProgram(program);

    // Check for errors
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(
        "Transform feedback program link error:",
        gl.getProgramInfoLog(program),
      );
      return false;
    }

    // Set up transform feedback
    const transformFeedback = gl.createTransformFeedback();
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, transformFeedback);

    // Bind target buffer for transform feedback
    gl.bindBuffer(gl.ARRAY_BUFFER, targetBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      options.vertexCount * options.components * 4,
      gl.STATIC_DRAW,
    );
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, targetBuffer);

    // Use program and set uniforms
    gl.useProgram(program);
    gl.uniform1i(gl.getUniformLocation(program, "u_texture"), 0);
    gl.uniform2f(
      gl.getUniformLocation(program, "u_textureSize"),
      options.width,
      options.height,
    );

    // Bind texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Run transform feedback (no vertex array needed for this approach)
    gl.enable(gl.RASTERIZER_DISCARD); // Don't render to screen
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, options.vertexCount);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    // Cleanup
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.deleteTransformFeedback(transformFeedback);
    gl.deleteProgram(program);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    return true;
  } catch (error) {
    console.error("Transform feedback error:", error);
    return false;
  }
}

export function CubeWithBuffers() {
  const meshRef = useRef<THREE.Mesh>(null);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const positionTensorRef = useRef<tf.Tensor | null>(null);

  const initializeTensorBuffers = async () => {
    console.log("=== Initializing tensor buffers ===");

    // Create tensor data - for example, vertex positions
    // Each row represents a vertex (x, y, z)
    const vertexData = tf.tensor2d(
      [
        [0, 0, 0], // vertex 0
        [1, 0, 0], // vertex 1
        [0, 1, 0], // vertex 2
        [0, 0, 1], // vertex 3
      ],
      [4, 3],
    );

    positionTensorRef.current = vertexData;

    // Get the WebGL backend
    const backend = tf.backend() as tf.MathBackendWebGL;
    const gl = backend.gpgpu.gl;

    if (backend) {
      // Upload tensor to GPU if not already there
      backend.uploadToGPU(vertexData.dataId);

      // Get the WebGL buffer from TensorFlow.js
      const textureData = backend.readToGPU(vertexData.dataId);
      console.log(textureData.texture);

      // Create a vertex buffer to hold the converted data
      const positionBuffer = gl.createBuffer();

      // Use transform feedback to convert texture to buffer
      const success = textureToBuffer(gl, textureData.texture, positionBuffer, {
        width: 1,
        height: 3,
        components: 3, // x, y, z
        vertexCount: 4,
      });

      if (success && positionBuffer) {
        console.log("Successfully converted texture to buffer");
        console.log(positionBuffer);

        // Create GLBufferAttribute from the converted buffer
        const positionAttribute = new THREE.GLBufferAttribute(
          positionBuffer, // WebGL buffer created from texture
          gl.FLOAT, // data type
          3, // items per vertex (x, y, z)
          4, // bytes per item (float32)
          4, // vertex count
        );

        const bufferGeometry = new THREE.BufferGeometry();
        bufferGeometry.setAttribute("position", positionAttribute);
        bufferGeometry.setDrawRange(0, 4);
        setGeometry(bufferGeometry);
        console.log("Successfully created geometry with converted buffer");
      }
    }
  };

  // Initialize tensor buffers
  useSharedWebGLContext<Promise<void>>(initializeTensorBuffers, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (positionTensorRef.current) {
        positionTensorRef.current.dispose();
      }
    };
  }, []);

  if (!geometry) {
    return null; // Wait for geometry to be ready
  }

  return (
    <points>
      <primitive object={geometry} />
      <pointsMaterial size={10} color="red" />
    </points>
  );
}
