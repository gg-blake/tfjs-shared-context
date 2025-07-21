import { registerKernel } from "@tensorflow/tfjs-core";
import { _fusedMatMulConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/_FusedMatMul";
import { absConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Abs";
import { acosConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Acos";
import { acoshConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Acosh";
import { addConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Add";
import { addNConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/AddN";
import { allConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/All";
import { anyConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Any";
import { argMaxConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ArgMax";
import { argMinConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ArgMin";
import { asinConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Asin";
import { asinhConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Asinh";
import { atanConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Atan";
import { atan2Config } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Atan2";
import { atanhConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Atanh";
import { avgPoolConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/AvgPool";
import { avgPool3DConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/AvgPool3D";
import { avgPool3DGradConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/AvgPool3DGrad";
import { avgPoolGradConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/AvgPoolGrad";
import { batchMatMulConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/BatchMatMul";
import { batchNormConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/BatchNorm";
import { batchToSpaceNDConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/BatchToSpaceND";
import { bincountConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Bincount";
import { bitwiseAndConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/BitwiseAnd";
import { broadcastArgsConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/BroadcastArgs";
import { castConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Cast";
import { ceilConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Ceil";
import { clipByValueConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ClipByValue";
import { complexConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Complex";
import { complexAbsConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ComplexAbs";
import { concatConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Concat";
import { conv2DConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Conv2D";
import { conv2DBackpropFilterConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Conv2DBackpropFilter";
import { conv2DBackpropInputConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Conv2DBackpropInput";
import { conv3DConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Conv3D";
import { conv3DBackpropFilterV2Config } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Conv3DBackpropFilterV2";
import { conv3DBackpropInputConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Conv3DBackpropInputV2";
import { cosConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Cos";
import { coshConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Cosh";
import { cropAndResizeConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/CropAndResize";
import { cumprodConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Cumprod";
import { cumsumConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Cumsum";
import { denseBincountConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/DenseBincount";
import { depthToSpaceConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/DepthToSpace";
import { depthwiseConv2dNativeConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/DepthwiseConv2dNative";
import { depthwiseConv2dNativeBackpropFilterConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/DepthwiseConv2dNativeBackpropFilter";
import { depthwiseConv2dNativeBackpropInputConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/DepthwiseConv2dNativeBackpropInput";
import { diagConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Diag";
import { dilation2DConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Dilation2D";
import { einsumConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Einsum";
import { eluConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Elu";
import { eluGradConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/EluGrad";
import { equalConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Equal";
import { erfConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Erf";
import { expConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Exp";
import { expandDimsConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ExpandDims";
import { expm1Config } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Expm1";
import { fftConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/FFT";
import { fillConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Fill";
import { flipLeftRightConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/FlipLeftRight";
import { floorConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Floor";
import { floorDivConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/FloorDiv";
import { fromPixelsConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/FromPixels";
import { fusedConv2DConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/FusedConv2D";
import { fusedDepthwiseConv2DConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/FusedDepthwiseConv2D";
import { gatherNdConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/GatherNd";
import { gatherV2Config } from "@tensorflow/tfjs-backend-webgl/dist/kernels/GatherV2";
import { greaterConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Greater";
import { greaterEqualConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/GreaterEqual";
import { identityConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Identity";
import { ifftConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/IFFT";
import { imagConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Imag";
import { isFiniteConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/IsFinite";
import { isInfConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/IsInf";
import { isNaNConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/IsNaN";
import { leakyReluConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/LeakyRelu";
import { lessConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Less";
import { lessEqualConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/LessEqual";
import { linSpaceConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/LinSpace";
import { logConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Log";
import { log1pConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Log1p";
import { logicalAndConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/LogicalAnd";
import { logicalNotConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/LogicalNot";
import { logicalOrConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/LogicalOr";
import { LRNConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/LRN";
import { LRNGradConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/LRNGrad";
import { maxConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Max";
import { maximumConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Maximum";
import { maxPoolConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/MaxPool";
import { maxPool3DConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/MaxPool3D";
import { maxPool3DGradConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/MaxPool3DGrad";
import { maxPoolGradConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/MaxPoolGrad";
import { maxPoolWithArgmaxConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/MaxPoolWithArgmax";
import { meanConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Mean";
import { minConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Min";
import { minimumConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Minimum";
import { mirrorPadConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/MirrorPad";
import { modConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Mod";
import { multinomialConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Multinomial";
import { multiplyConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Multiply";
import { negConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Neg";
import { nonMaxSuppressionV3Config } from "@tensorflow/tfjs-backend-webgl/dist/kernels/NonMaxSuppressionV3";
import { nonMaxSuppressionV4Config } from "@tensorflow/tfjs-backend-webgl/dist/kernels/NonMaxSuppressionV4";
import { nonMaxSuppressionV5Config } from "@tensorflow/tfjs-backend-webgl/dist/kernels/NonMaxSuppressionV5";
import { notEqualConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/NotEqual";
import { oneHotConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/OneHot";
import { onesLikeConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/OnesLike";
import { packConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Pack";
import { padV2Config } from "@tensorflow/tfjs-backend-webgl/dist/kernels/PadV2";
import { powConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Pow";
import { preluConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Prelu";
import { prodConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Prod";
import { raggedGatherConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/RaggedGather";
import { raggedRangeConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/RaggedRange";
import { raggedTensorToTensorConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/RaggedTensorToTensor";
import { rangeConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Range";
import { realConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Real";
import { realDivConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/RealDiv";
import { reciprocalConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Reciprocal";
import { reluConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Relu";
import { relu6Config } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Relu6";
import { reshapeConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Reshape";
import { resizeBilinearConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ResizeBilinear";
import { resizeBilinearGradConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ResizeBilinearGrad";
import { resizeNearestNeighborConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ResizeNearestNeighbor";
import { resizeNearestNeighborGradConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ResizeNearestNeighborGrad";
import { reverseConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Reverse";
import { rotateWithOffsetConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/RotateWithOffset";
import { roundConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Round";
import { rsqrtConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Rsqrt";
import { scatterNdConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ScatterNd";
import { searchSortedConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/SearchSorted";
import { selectConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Select";
import { seluConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Selu";
import { sigmoidConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Sigmoid";
import { signConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Sign";
import { sinConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Sin";
import { sinhConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Sinh";
import { sliceConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Slice";
import { softmaxConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Softmax";
import { softplusConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Softplus";
import { spaceToBatchNDConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/SpaceToBatchND";
import { sparseFillEmptyRowsConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/SparseFillEmptyRows";
import { sparseReshapeConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/SparseReshape";
import { sparseSegmentMeanConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/SparseSegmentMean";
import { sparseSegmentSumConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/SparseSegmentSum";
import { sparseToDenseConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/SparseToDense";
import { splitVConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/SplitV";
import { sqrtConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Sqrt";
import { squareConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Square";
import { squaredDifferenceConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/SquaredDifference";
import { staticRegexReplaceConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/StaticRegexReplace";
import { stepConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Step";
import { stridedSliceConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/StridedSlice";
import { stringNGramsConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/StringNGrams";
import { stringSplitConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/StringSplit";
import { stringToHashBucketFastConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/StringToHashBucketFast";
import { subConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Sub";
import { sumConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Sum";
import { tanConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Tan";
import { tanhConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Tanh";
import { tensorScatterUpdateConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/TensorScatterUpdate";
import { tileConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Tile";
import { topKConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/TopK";
import { transformConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Transform";
import { transposeConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Transpose";
import { uniqueConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Unique";
import { unpackConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/Unpack";
import { unsortedSegmentSumConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/UnsortedSegmentSum";
import { zerosLikeConfig } from "@tensorflow/tfjs-backend-webgl/dist/kernels/ZerosLike";
// List all kernel configs here
export const kernelConfigs = [
  _fusedMatMulConfig,
  absConfig,
  acosConfig,
  acoshConfig,
  addConfig,
  addNConfig,
  allConfig,
  anyConfig,
  argMaxConfig,
  argMinConfig,
  asinConfig,
  asinhConfig,
  atanConfig,
  atan2Config,
  atanhConfig,
  avgPoolConfig,
  avgPool3DConfig,
  avgPool3DGradConfig,
  avgPoolGradConfig,
  batchMatMulConfig,
  batchNormConfig,
  batchToSpaceNDConfig,
  bincountConfig,
  bitwiseAndConfig,
  broadcastArgsConfig,
  castConfig,
  ceilConfig,
  clipByValueConfig,
  complexConfig,
  complexAbsConfig,
  concatConfig,
  conv2DConfig,
  conv2DBackpropFilterConfig,
  conv2DBackpropInputConfig,
  conv3DConfig,
  conv3DBackpropFilterV2Config,
  conv3DBackpropInputConfig,
  cosConfig,
  coshConfig,
  cropAndResizeConfig,
  cumprodConfig,
  cumsumConfig,
  denseBincountConfig,
  depthToSpaceConfig,
  depthwiseConv2dNativeConfig,
  depthwiseConv2dNativeBackpropFilterConfig,
  depthwiseConv2dNativeBackpropInputConfig,
  diagConfig,
  dilation2DConfig,
  einsumConfig,
  eluConfig,
  eluGradConfig,
  equalConfig,
  erfConfig,
  expConfig,
  expandDimsConfig,
  expm1Config,
  fftConfig,
  fillConfig,
  flipLeftRightConfig,
  floorConfig,
  floorDivConfig,
  fromPixelsConfig,
  fusedConv2DConfig,
  fusedDepthwiseConv2DConfig,
  gatherNdConfig,
  gatherV2Config,
  greaterConfig,
  greaterEqualConfig,
  identityConfig,
  ifftConfig,
  imagConfig,
  isFiniteConfig,
  isInfConfig,
  isNaNConfig,
  leakyReluConfig,
  lessConfig,
  lessEqualConfig,
  linSpaceConfig,
  logConfig,
  log1pConfig,
  logicalAndConfig,
  logicalNotConfig,
  logicalOrConfig,
  LRNConfig,
  LRNGradConfig,
  maxConfig,
  maximumConfig,
  maxPoolConfig,
  maxPool3DConfig,
  maxPool3DGradConfig,
  maxPoolGradConfig,
  maxPoolWithArgmaxConfig,
  meanConfig,
  minConfig,
  minimumConfig,
  mirrorPadConfig,
  modConfig,
  multinomialConfig,
  multiplyConfig,
  negConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  oneHotConfig,
  onesLikeConfig,
  packConfig,
  padV2Config,
  powConfig,
  preluConfig,
  prodConfig,
  raggedGatherConfig,
  raggedRangeConfig,
  raggedTensorToTensorConfig,
  rangeConfig,
  realConfig,
  realDivConfig,
  reciprocalConfig,
  reluConfig,
  relu6Config,
  reshapeConfig,
  resizeBilinearConfig,
  resizeBilinearGradConfig,
  resizeNearestNeighborConfig,
  resizeNearestNeighborGradConfig,
  reverseConfig,
  rotateWithOffsetConfig,
  roundConfig,
  rsqrtConfig,
  scatterNdConfig,
  searchSortedConfig,
  selectConfig,
  seluConfig,
  sigmoidConfig,
  signConfig,
  sinConfig,
  sinhConfig,
  sliceConfig,
  softmaxConfig,
  softplusConfig,
  spaceToBatchNDConfig,
  sparseFillEmptyRowsConfig,
  sparseReshapeConfig,
  sparseSegmentMeanConfig,
  sparseSegmentSumConfig,
  sparseToDenseConfig,
  splitVConfig,
  sqrtConfig,
  squareConfig,
  squaredDifferenceConfig,
  staticRegexReplaceConfig,
  stepConfig,
  stridedSliceConfig,
  stringNGramsConfig,
  stringSplitConfig,
  stringToHashBucketFastConfig,
  subConfig,
  sumConfig,
  tanConfig,
  tanhConfig,
  tensorScatterUpdateConfig,
  tileConfig,
  topKConfig,
  transformConfig,
  transposeConfig,
  uniqueConfig,
  unpackConfig,
  unsortedSegmentSumConfig,
  zerosLikeConfig,
];
