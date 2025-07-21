"use client";

import { useRef, useEffect, useState } from "react";
import { useFrame, useThree, RootState } from "@react-three/fiber";
import * as THREE from "three";
import * as tf from "@tensorflow/tfjs";
import { useSharedWebGLContext } from "@/helpers/hooks/useSharedWebGLContext";
import {
  createTestCanvas,
  createTestCanvasTexture,
  createTextureFromTensor,
  tensorToTexture
} from "@/helpers/canvasToTensor";

// Vertex shader that reads position from TensorFlow.js texture
const vertexShader = `
  precision highp float;
  uniform sampler2D positionTexture;
  uniform float time;
  varying vec3 vNormal;
  varying vec3 vPosition;
  

  void main() {
    vec4 pos = texture2D(positionTexture, vec2(0.5, 0.5));
    vNormal = normalize(normalMatrix * normal);
    vec4 worldPosition = modelViewMatrix * vec4(position + pos.xyz, 1.0);
    vPosition = worldPosition.xyz;
    gl_Position = projectionMatrix * worldPosition;
  }
`;

// Simple blue fragment shader
const fragmentShader = `
  uniform vec3 uLightPos;
  uniform vec3 uColor;
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform sampler2D positionTexture;
  
  void main() {
    vec4 color = texture2D(positionTexture, vec2(0.5, 0.5));
    vec3 lightDir = normalize(uLightPos - vPosition);
    float diff = max(dot(vNormal, lightDir), 0.0);
    vec3 diffuse = diff * color.xyz;
    gl_FragColor = vec4(diffuse, 1.0);
  }
`;

export function Cube() {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  // Create a blank canvas for its texture (TODO: try to make a texture without using a canvas)
  const tensorTexture = useRef<THREE.Texture | null>(createTestCanvasTexture());
  const canvasRef = useRef<HTMLCanvasElement>(createTestCanvas())
  const positionTensorRef = useRef<tf.Variable | null>(null);
  const state = useThree();

  const initializeTensor = async () => {
    // Create a tensor for position (x, y, z)
    // Initialize at origin, but you can update this with your model output
    positionTensorRef.current = tf.variable(tf.tensor([1.0, 0.0, 0.0, 1.0], [1, 4], "float32"));
    state.gl.initTexture(tensorTexture.current);
    const texture = tensorToTexture(state, positionTensorRef.current, tensorTexture.current, true)
    tensorTexture.current = texture;
  };

  // Initialize TensorFlow.js tensor and extract WebGL texture
  useSharedWebGLContext<Promise<void>>(initializeTensor, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (positionTensorRef.current) {
        positionTensorRef.current.dispose();
      }
    };
  }, []);

  // Animation loop
  useFrame((state) => {
    if (materialRef.current) {
      
      
      materialRef.current.needsUpdate = true;
      materialRef.current.uniforms.positionTexture.value = tensorTexture.current;
      materialRef.current.uniforms.time.value = state.clock.getElapsedTime();
    }
  });

  return (
    <mesh ref={meshRef}>
      <boxGeometry args={[1, 1, 1]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={{
          positionTexture: { value: tensorTexture.current },
          time: { value: 0 },
          uLightPos: { value: new THREE.Vector3(10, 10, 10) },
          uColor: { value: new THREE.Color('orange') }
        }}
      />
    </mesh>
  );
}
