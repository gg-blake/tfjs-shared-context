"use client";

import { Canvas } from "@react-three/fiber";
import { Preload } from "@react-three/drei";
import { r3f } from "@/helpers/global";
import * as THREE from "three";
import { useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { kernelConfigs } from "@/helpers/register";

export default function Scene({ ...props }) {
  // Everything defined in here will persist between route changes, only children are swapped
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleCanvasCreated = async (state) => {
    state.gl.toneMapping = THREE.AgXToneMapping;

    // Share canvas context with tensorflowjs
    if (tf.findBackend("custom-webgl") == null) {
      if (!canvasRef.current) return;
      const customCanvas = canvasRef.current;
      const customBackend = new tf.MathBackendWebGL(customCanvas);
      tf.registerBackend("custom-webgl", () => customBackend, 1);

      // Register all kernels under shared webgl context/backend
      for (const kernelConfig of kernelConfigs) {
        tf.registerKernel({
          ...kernelConfig,
          backendName: "custom-webgl",
        });
      }
    }
    await tf.setBackend("custom-webgl");
    console.log(tf.getBackend());
  };

  return (
    <Canvas ref={canvasRef} {...props} onCreated={handleCanvasCreated}>
      <r3f.Out />
      <Preload all />
    </Canvas>
  );
}
