import { useEffect } from "react";
import { useSharedWebGLContext } from "@/helpers/hooks/useSharedWebGLContext";


export function Cube(props) {
  useSharedWebGLContext(() => {
    // Animation and position logic goes here
    ...
  });
  
  return (
    <mesh>
      <boxGeometry args={...} />
      <shaderMaterial
        vertexShader={...}
        fragmentShader={...}
        uniforms={{
          positionTexture: { value: ... },
          time: { value: 0 },
        }}
      />
    </mesh>
  )
}