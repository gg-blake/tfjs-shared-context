import { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

export const useSharedWebGLContext = <T>(callbackFn: () => T, dependencies) => {
  useEffect(() => {
    waitForContext(callbackFn);
  }, dependencies);

  const waitForContext = async (callbackFn: () => T) => {
    await tf.ready();
    callbackFn();
  };
};
