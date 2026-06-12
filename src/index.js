import {
  FaceLandmarker,
  FilesetResolver,
  ImageSegmenter
} from "@mediapipe/tasks-vision";

const DEFAULT_WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const DEFAULT_FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const DEFAULT_SEGMENTER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";

const FACE_OVAL_LANDMARK_INDEXES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
  378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
  162, 21, 54, 103, 67, 109
];

/**
 * @typedef {object} CutoutTasks
 * @property {ImageSegmenter} segmenter
 * @property {FaceLandmarker} faceLandmarker
 */

/**
 * @typedef {HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap} CutoutImageSource
 */

/**
 * @typedef {object} AlphaMaskInput
 * @property {Uint8Array} mask
 * @property {number} maskWidth
 * @property {number} maskHeight
 * @property {number} width
 * @property {number} height
 * @property {Set<number>} keepCategoryIndexes
 * @property {Set<number>} accessoryCategoryIndexes
 * @property {{minX: number, minY: number, maxX: number, maxY: number}} faceRegion
 * @property {number} accessoryFaceExpansionRatio
 * @property {number} chinY
 * @property {number} faceHeight
 * @property {number} chinMarginRatio
 * @property {number} bottomFeatherRatio
 */

/**
 * Creates the MediaPipe tasks used by the cutout pipeline.
 *
 * By default this uses MediaPipe's Multi-class Selfie Segmenter:
 * 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 accessories.
 *
 * @param {object} [options]
 * @param {string} [options.wasmRoot]
 * @param {string} [options.faceModelPath]
 * @param {string} [options.segmenterModelPath]
 * @returns {Promise<CutoutTasks>}
 */
export async function createCutoutTasks(options = {}) {
  const {
    wasmRoot = DEFAULT_WASM_ROOT,
    faceModelPath = DEFAULT_FACE_MODEL,
    segmenterModelPath = DEFAULT_SEGMENTER_MODEL
  } = options;

  const fileset = await FilesetResolver.forVisionTasks(wasmRoot);
  const [segmenter, faceLandmarker] = await Promise.all([
    ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: segmenterModelPath },
      runningMode: "IMAGE",
      outputCategoryMask: true
    }),
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: faceModelPath },
      runningMode: "IMAGE",
      numFaces: 1
    })
  ]);

  return { segmenter, faceLandmarker };
}

/**
 * Produces a transparent PNG cutout using segmentation for the external shape
 * and face landmarks only for the bottom cutoff.
 *
 * @param {CutoutImageSource} image
 * @param {CutoutTasks} tasks
 * @param {object} [options]
 * @param {number[]} [options.keepCategoryIndexes]
 * @param {number[]} [options.accessoryCategoryIndexes]
 * @param {number} [options.accessoryFaceExpansionRatio]
 * @param {number} [options.maxWidth]
 * @param {number} [options.maxHeight]
 * @param {number} [options.topPaddingRatio]
 * @param {number} [options.sidePaddingRatio]
 * @param {number} [options.bottomPaddingRatio]
 * @param {number} [options.inputScaleRatio]
 * @param {"contain"} [options.outputFit]
 * @param {number} [options.outputTrimTopRatio]
 * @param {number} [options.outputTrimBottomRatio]
 * @param {number} [options.cropPaddingRatio]
 * @param {number} [options.cropPaddingXRatio]
 * @param {number} [options.cropPaddingTopRatio]
 * @param {number} [options.cropPaddingBottomRatio]
 * @param {boolean} [options.debugCrop]
 * @param {number} [options.chinMarginRatio]
 * @param {number} [options.bottomFeatherRatio]
 * @param {number} [options.maskFeatherPx]
 * @returns {Promise<{canvas: HTMLCanvasElement, pngBlob: Blob, pngDataUrl: string, bounds: {x: number, y: number, width: number, height: number}, rawMaskBox: {x: number, y: number, width: number, height: number}, paddedCropBox: {x: number, y: number, width: number, height: number}, requestedCropBox: {x: number, y: number, width: number, height: number}, cropCanvasSize: {width: number, height: number}, outputSize: {width: number, height: number}, resize: {fit: "contain", scale: number, sourceWidth: number, sourceHeight: number, outputWidth: number, outputHeight: number}, outputTrim: {top: number, bottom: number, sourceWidth: number, sourceHeight: number, outputWidth: number, outputHeight: number}, debug?: {segmentationMaskCanvas: HTMLCanvasElement, finalAlphaMaskCanvas: HTMLCanvasElement, cropCanvas: HTMLCanvasElement}, debugCanvas?: HTMLCanvasElement}>}
 */
export async function cutoutFaceHair(image, tasks, options = {}) {
  const {
    keepCategoryIndexes = [1, 3],
    accessoryCategoryIndexes = [5],
    accessoryFaceExpansionRatio = 0.18,
    maxWidth = 1024,
    maxHeight = 1024,
    cropPaddingRatio = 0.18,
    cropPaddingXRatio,
    cropPaddingTopRatio,
    cropPaddingBottomRatio,
    topPaddingRatio = cropPaddingTopRatio ?? 0.25,
    sidePaddingRatio = cropPaddingXRatio ?? 0.14,
    bottomPaddingRatio = cropPaddingBottomRatio ?? 0.06,
    inputScaleRatio = 0.75,
    outputFit = "contain",
    outputTrimTopRatio = 0,
    outputTrimBottomRatio = 0,
    debugCrop = false,
    chinMarginRatio = 0.06,
    bottomFeatherRatio = 0.02,
    maskFeatherPx = 1
  } = options;

  if (outputFit !== "contain") {
    throw new Error('Unsupported outputFit. Use "contain" to resize without cropping.');
  }

  const source = drawToCanvas(image, inputScaleRatio);
  const width = source.width;
  const height = source.height;
  const segmentation = tasks.segmenter.segment(source);
  const landmarks = tasks.faceLandmarker.detect(source);
  const face = landmarks.faceLandmarks?.[0];

  if (!face) {
    throw new Error("No face detected.");
  }

  const categoryMask = segmentation.categoryMask;
  if (!categoryMask) {
    throw new Error("Image Segmenter did not return a category mask.");
  }

  const maskWidth = categoryMask.width;
  const maskHeight = categoryMask.height;
  const mask = readMask(categoryMask);
  categoryMask.close();
  const faceMetrics = getFaceMetrics(face, width, height);
  const segmentationAlpha = buildSegmentationMask({
    mask,
    maskWidth,
    maskHeight,
    width,
    height,
    keepCategoryIndexes: new Set(keepCategoryIndexes),
    accessoryCategoryIndexes: new Set(accessoryCategoryIndexes),
    faceRegion: faceMetrics.region,
    accessoryFaceExpansionRatio
  });
  const alpha = buildAlphaMask({
    mask,
    maskWidth,
    maskHeight,
    width,
    height,
    keepCategoryIndexes: new Set(keepCategoryIndexes),
    accessoryCategoryIndexes: new Set(accessoryCategoryIndexes),
    faceRegion: faceMetrics.region,
    accessoryFaceExpansionRatio,
    chinY: faceMetrics.chinY,
    faceHeight: faceMetrics.faceHeight,
    chinMarginRatio,
    bottomFeatherRatio
  });

  const refinedAlpha = maskFeatherPx > 0 ? featherAlpha(alpha, width, height, maskFeatherPx) : alpha;
  const rawMaskBox = getAlphaBounds(refinedAlpha, width, height);
  if (!rawMaskBox) {
    throw new Error("The final mask is empty. Check segmenter category indexes.");
  }

  const cropLayout = getCropLayout(rawMaskBox, width, height, {
    topRatio: topPaddingRatio,
    sideRatio: sidePaddingRatio,
    bottomRatio: bottomPaddingRatio
  });
  const cropped = cropSourceWithAlpha(source, refinedAlpha, cropLayout);
  const { canvas: resized, resize } = resizeToFit(cropped, maxWidth, maxHeight);
  const { canvas: output, outputTrim } = trimVerticalOutput(
    resized,
    outputTrimTopRatio,
    outputTrimBottomRatio
  );
  const outputSize = { width: output.width, height: output.height };
  const debug = debugCrop
    ? {
        segmentationMaskCanvas: drawAlphaMask(segmentationAlpha, width, height),
        finalAlphaMaskCanvas: drawAlphaMask(refinedAlpha, width, height),
        cropCanvas: drawDebugCrop(
          source,
          rawMaskBox,
          cropLayout.paddedCropBox,
          faceMetrics.chinY + faceMetrics.faceHeight * chinMarginRatio
        )
      }
    : undefined;

  if (debugCrop) {
    console.log("face-hair-cutout debug", {
      segmentationMaskBox: getAlphaBounds(segmentationAlpha, width, height),
      finalMaskBox: rawMaskBox,
      rawMaskBox,
      paddedCropBox: cropLayout.paddedCropBox,
      requestedCropBox: cropLayout.requestedCropBox,
      cropCanvasSize: {
        width: cropLayout.outputWidth,
        height: cropLayout.outputHeight
      },
      drawOffset: {
        x: cropLayout.drawX,
        y: cropLayout.drawY
      },
      outputSize,
      resize,
      outputTrim
    });
  }

  const pngBlob = await canvasToBlob(output);

  return {
    canvas: output,
    pngBlob,
    pngDataUrl: output.toDataURL("image/png"),
    bounds: cropLayout.paddedCropBox,
    rawMaskBox,
    paddedCropBox: cropLayout.paddedCropBox,
    requestedCropBox: cropLayout.requestedCropBox,
    cropCanvasSize: {
      width: cropLayout.outputWidth,
      height: cropLayout.outputHeight
    },
    outputSize,
    resize,
    outputTrim,
    ...(debug ? { debug, debugCanvas: debug.cropCanvas } : {})
  };
}

/**
 * @param {CutoutImageSource} image
 * @param {number} inputScaleRatio
 * @returns {HTMLCanvasElement}
 */
function drawToCanvas(image, inputScaleRatio) {
  const width = Math.round(getSourceWidth(image));
  const height = Math.round(getSourceHeight(image));
  const canvas = makeCanvas(width, height);
  const scale = clamp(inputScaleRatio, 0.1, 1);
  const drawWidth = Math.round(width * scale);
  const drawHeight = Math.round(height * scale);
  const drawX = Math.round((width - drawWidth) / 2);
  const drawY = Math.round((height - drawHeight) / 2);
  const context = get2d(canvas);
  context.clearRect(0, 0, width, height);
  context.drawImage(image, drawX, drawY, drawWidth, drawHeight);
  return canvas;
}

/**
 * @param {import("@mediapipe/tasks-vision").MPMask} mask
 * @returns {Uint8Array}
 */
function readMask(mask) {
  if (typeof mask.getAsUint8Array === "function") {
    return mask.getAsUint8Array();
  }

  if (typeof mask.getAsFloat32Array === "function") {
    const floatData = mask.getAsFloat32Array();
    const data = new Uint8Array(floatData.length);
    for (let i = 0; i < floatData.length; i += 1) data[i] = Math.round(floatData[i]);
    return data;
  }

  throw new Error("Unsupported MediaPipe mask type.");
}

/**
 * @param {Omit<AlphaMaskInput, "chinY" | "faceHeight" | "chinMarginRatio" | "bottomFeatherRatio">} input
 * @returns {Uint8ClampedArray}
 */
function buildSegmentationMask({
  mask,
  maskWidth,
  maskHeight,
  width,
  height,
  keepCategoryIndexes,
  accessoryCategoryIndexes,
  faceRegion,
  accessoryFaceExpansionRatio
}) {
  const alpha = new Uint8ClampedArray(width * height);
  const accessoryRegion = expandRegion(
    faceRegion,
    Math.max(1, faceRegion.maxY - faceRegion.minY) * accessoryFaceExpansionRatio,
    width,
    height
  );

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const maskX = Math.min(maskWidth - 1, Math.floor((x / width) * maskWidth));
      const maskY = Math.min(maskHeight - 1, Math.floor((y / height) * maskHeight));
      const category = mask[maskY * maskWidth + maskX];
      if (
        keepCategoryIndexes.has(category) ||
        (accessoryCategoryIndexes.has(category) && isInsideRegion(x, y, accessoryRegion))
      ) {
        alpha[y * width + x] = 255;
      }
    }
  }

  return alpha;
}

/**
 * @param {import("@mediapipe/tasks-vision").NormalizedLandmark[]} face
 * @param {number} width
 * @param {number} height
 */
function getFaceMetrics(face, width, height) {
  const oval = FACE_OVAL_LANDMARK_INDEXES.map((index) => face[index]).filter(Boolean);
  const xs = oval.map((point) => point.x * width);
  const ys = oval.map((point) => point.y * height);
  const chinY = Math.max(...ys);
  const topY = Math.min(...ys);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  return {
    chinY,
    faceHeight: Math.max(1, chinY - topY),
    faceWidth: Math.max(1, maxX - minX),
    region: { minX, minY: topY, maxX, maxY: chinY }
  };
}

/**
 * @param {AlphaMaskInput} input
 * @returns {Uint8ClampedArray}
 */
function buildAlphaMask({
  mask,
  maskWidth,
  maskHeight,
  width,
  height,
  keepCategoryIndexes,
  accessoryCategoryIndexes,
  faceRegion,
  accessoryFaceExpansionRatio,
  chinY,
  faceHeight,
  chinMarginRatio,
  bottomFeatherRatio
}) {
  const alpha = new Uint8ClampedArray(width * height);
  const margin = faceHeight * chinMarginRatio;
  const feather = Math.max(1, faceHeight * bottomFeatherRatio);
  const centerX = width / 2;
  const curveDepth = faceHeight * 0.025;
  const accessoryRegion = expandRegion(
    faceRegion,
    faceHeight * accessoryFaceExpansionRatio,
    width,
    height
  );
  const headAccessoryRegion = expandRegion(faceRegion, faceHeight * 0.75, width, height);
  headAccessoryRegion.maxY = Math.min(
    height - 1,
    chinY + margin + feather
  );

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      const maskX = Math.min(maskWidth - 1, Math.floor((x / width) * maskWidth));
      const maskY = Math.min(maskHeight - 1, Math.floor((y / height) * maskHeight));
      const category = mask[maskY * maskWidth + maskX];
      const keepMainCategory = keepCategoryIndexes.has(category);
      const keepFaceAccessory =
        accessoryCategoryIndexes.has(category) &&
        (isInsideRegion(x, y, accessoryRegion) || isInsideRegion(x, y, headAccessoryRegion));
      if (!keepMainCategory && !keepFaceAccessory) continue;

      const normalizedX = Math.abs((x - centerX) / Math.max(1, width / 2));
      const curvedCutY = chinY + margin + curveDepth * (1 - normalizedX * normalizedX);
      const distanceBelowCut = y - curvedCutY;
      if (distanceBelowCut <= 0) {
        alpha[i] = 255;
      } else if (distanceBelowCut < feather) {
        alpha[i] = Math.round(255 * (1 - distanceBelowCut / feather));
      }
    }
  }

  return alpha;
}

/**
 * @param {{minX: number, minY: number, maxX: number, maxY: number}} region
 * @param {number} amount
 * @param {number} width
 * @param {number} height
 */
function expandRegion(region, amount, width, height) {
  return {
    minX: Math.max(0, region.minX - amount),
    minY: Math.max(0, region.minY - amount),
    maxX: Math.min(width - 1, region.maxX + amount),
    maxY: Math.min(height - 1, region.maxY + amount)
  };
}

/**
 * @param {number} x
 * @param {number} y
 * @param {{minX: number, minY: number, maxX: number, maxY: number}} region
 */
function isInsideRegion(x, y, region) {
  return x >= region.minX && x <= region.maxX && y >= region.minY && y <= region.maxY;
}

/**
 * @param {Uint8ClampedArray} alpha
 * @param {number} width
 * @param {number} height
 * @param {number} radius
 * @returns {Uint8ClampedArray}
 */
function featherAlpha(alpha, width, height, radius) {
  let current = alpha;
  for (let pass = 0; pass < radius; pass += 1) {
    const next = new Uint8ClampedArray(current.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let sum = 0;
        let count = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            sum += current[ny * width + nx];
            count += 1;
          }
        }
        next[y * width + x] = Math.round(sum / count);
      }
    }
    current = next;
  }
  return current;
}

/**
 * @param {Uint8ClampedArray} alpha
 * @param {number} width
 * @param {number} height
 * @returns {{x: number, y: number, width: number, height: number} | null}
 */
function getAlphaBounds(alpha, width, height) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (alpha[y * width + x] === 0) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  if (maxX < minX || maxY < minY) return null;
  return {
    x: minX,
    y: minY,
    width: maxX - minX + 1,
    height: maxY - minY + 1
  };
}

/**
 * @param {number} width
 * @param {number} height
 * @returns {HTMLCanvasElement}
 */
function makeCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/**
 * @param {{x: number, y: number, width: number, height: number}} rawMaskBox
 * @param {number} sourceWidth
 * @param {number} sourceHeight
 * @param {{topRatio: number, sideRatio: number, bottomRatio: number}} padding
 * @returns {{paddedCropBox: {x: number, y: number, width: number, height: number}, requestedCropBox: {x: number, y: number, width: number, height: number}, outputWidth: number, outputHeight: number, drawX: number, drawY: number}}
 */
function getCropLayout(rawMaskBox, sourceWidth, sourceHeight, padding) {
  const top = Math.round(rawMaskBox.height * padding.topRatio);
  const side = Math.round(rawMaskBox.width * padding.sideRatio);
  const bottom = Math.round(rawMaskBox.height * padding.bottomRatio);
  const requestedX1 = rawMaskBox.x - side;
  const requestedY1 = rawMaskBox.y - top;
  const requestedX2 = rawMaskBox.x + rawMaskBox.width + side;
  const requestedY2 = rawMaskBox.y + rawMaskBox.height + bottom;
  const x1 = clamp(requestedX1, 0, sourceWidth - 1);
  const y1 = clamp(requestedY1, 0, sourceHeight - 1);
  const x2 = clamp(requestedX2, x1 + 1, sourceWidth);
  const y2 = clamp(requestedY2, y1 + 1, sourceHeight);
  const outputWidth = Math.max(1, Math.round(requestedX2 - requestedX1));
  const outputHeight = Math.max(1, Math.round(requestedY2 - requestedY1));

  return {
    requestedCropBox: {
      x: Math.round(requestedX1),
      y: Math.round(requestedY1),
      width: outputWidth,
      height: outputHeight
    },
    paddedCropBox: {
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1
    },
    outputWidth,
    outputHeight,
    drawX: Math.round(x1 - requestedX1),
    drawY: Math.round(y1 - requestedY1)
  };
}

/**
 * @param {HTMLCanvasElement} source
 * @param {Uint8ClampedArray} alpha
 * @param {{paddedCropBox: {x: number, y: number, width: number, height: number}, outputWidth: number, outputHeight: number, drawX: number, drawY: number}} layout
 * @returns {HTMLCanvasElement}
 */
function cropSourceWithAlpha(source, alpha, layout) {
  const { paddedCropBox: cropBox } = layout;
  const output = makeCanvas(layout.outputWidth, layout.outputHeight);
  const context = get2d(output);
  context.drawImage(
    source,
    cropBox.x,
    cropBox.y,
    cropBox.width,
    cropBox.height,
    layout.drawX,
    layout.drawY,
    cropBox.width,
    cropBox.height
  );

  const imageData = context.getImageData(0, 0, output.width, output.height);
  for (let y = 0; y < cropBox.height; y += 1) {
    for (let x = 0; x < cropBox.width; x += 1) {
      const sourceX = cropBox.x + x;
      const sourceY = cropBox.y + y;
      const outputX = layout.drawX + x;
      const outputY = layout.drawY + y;
      imageData.data[(outputY * output.width + outputX) * 4 + 3] =
        alpha[sourceY * source.width + sourceX];
    }
  }
  context.putImageData(imageData, 0, 0);
  return output;
}

/**
 * @param {HTMLCanvasElement} source
 * @param {number} maxWidth
 * @param {number} maxHeight
 * @returns {{canvas: HTMLCanvasElement, resize: {fit: "contain", scale: number, sourceWidth: number, sourceHeight: number, outputWidth: number, outputHeight: number}}}
 */
function resizeToFit(source, maxWidth, maxHeight) {
  const scale = Math.min(1, maxWidth / source.width, maxHeight / source.height);
  if (scale === 1) {
    return {
      canvas: source,
      resize: {
        fit: "contain",
        scale,
        sourceWidth: source.width,
        sourceHeight: source.height,
        outputWidth: source.width,
        outputHeight: source.height
      }
    };
  }

  const output = makeCanvas(
    Math.max(1, Math.round(source.width * scale)),
    Math.max(1, Math.round(source.height * scale))
  );
  get2d(output).drawImage(source, 0, 0, output.width, output.height);
  return {
    canvas: output,
    resize: {
      fit: "contain",
      scale,
      sourceWidth: source.width,
      sourceHeight: source.height,
      outputWidth: output.width,
      outputHeight: output.height
    }
  };
}

/**
 * @param {HTMLCanvasElement} source
 * @param {number} topRatio
 * @param {number} bottomRatio
 * @returns {{canvas: HTMLCanvasElement, outputTrim: {top: number, bottom: number, sourceWidth: number, sourceHeight: number, outputWidth: number, outputHeight: number}}}
 */
function trimVerticalOutput(source, topRatio, bottomRatio) {
  const top = Math.round(source.height * clamp(topRatio, 0, 0.45));
  const bottom = Math.round(source.height * clamp(bottomRatio, 0, 0.45));
  const outputHeight = Math.max(1, source.height - top - bottom);

  if (top === 0 && bottom === 0) {
    return {
      canvas: source,
      outputTrim: {
        top,
        bottom,
        sourceWidth: source.width,
        sourceHeight: source.height,
        outputWidth: source.width,
        outputHeight: source.height
      }
    };
  }

  const output = makeCanvas(source.width, outputHeight);
  get2d(output).drawImage(
    source,
    0,
    top,
    source.width,
    outputHeight,
    0,
    0,
    source.width,
    outputHeight
  );

  return {
    canvas: output,
    outputTrim: {
      top,
      bottom,
      sourceWidth: source.width,
      sourceHeight: source.height,
      outputWidth: output.width,
      outputHeight: output.height
    }
  };
}

/**
 * @param {HTMLCanvasElement} source
 * @param {{x: number, y: number, width: number, height: number}} rawMaskBox
 * @param {{x: number, y: number, width: number, height: number}} paddedCropBox
 * @param {number} chinCutY
 * @returns {HTMLCanvasElement}
 */
function drawDebugCrop(source, rawMaskBox, paddedCropBox, chinCutY) {
  const debugCanvas = makeCanvas(source.width, source.height);
  const context = get2d(debugCanvas);
  context.drawImage(source, 0, 0);
  drawBox(context, rawMaskBox, "#ef4444");
  drawBox(context, paddedCropBox, "#22c55e");
  drawLine(context, chinCutY, "#2563eb");
  return debugCanvas;
}

/**
 * @param {Uint8ClampedArray} alpha
 * @param {number} width
 * @param {number} height
 * @returns {HTMLCanvasElement}
 */
function drawAlphaMask(alpha, width, height) {
  const canvas = makeCanvas(width, height);
  const context = get2d(canvas);
  const imageData = context.createImageData(width, height);
  for (let i = 0; i < alpha.length; i += 1) {
    const value = alpha[i];
    imageData.data[i * 4] = value;
    imageData.data[i * 4 + 1] = value;
    imageData.data[i * 4 + 2] = value;
    imageData.data[i * 4 + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return canvas;
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {{x: number, y: number, width: number, height: number}} box
 * @param {string} color
 */
function drawBox(context, box, color) {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, Math.round(Math.min(context.canvas.width, context.canvas.height) * 0.004));
  context.strokeRect(box.x, box.y, box.width, box.height);
  context.restore();
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {number} y
 * @param {string} color
 */
function drawLine(context, y, color) {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, Math.round(Math.min(context.canvas.width, context.canvas.height) * 0.004));
  context.beginPath();
  context.moveTo(0, Math.round(y));
  context.lineTo(context.canvas.width, Math.round(y));
  context.stroke();
  context.restore();
}

/**
 * @param {number} value
 * @param {number} min
 * @param {number} max
 */
function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/**
 * @param {HTMLCanvasElement} canvas
 * @returns {Promise<Blob>}
 */
function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to encode PNG."));
    }, "image/png");
  });
}

/**
 * @param {HTMLCanvasElement} canvas
 * @returns {CanvasRenderingContext2D}
 */
function get2d(canvas) {
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D canvas is unavailable.");
  return context;
}

/**
 * @param {CutoutImageSource} image
 */
function getSourceWidth(image) {
  if (image instanceof HTMLImageElement) return image.naturalWidth;
  if (image instanceof HTMLVideoElement) return image.videoWidth;
  return image.width;
}

/**
 * @param {CutoutImageSource} image
 */
function getSourceHeight(image) {
  if (image instanceof HTMLImageElement) return image.naturalHeight;
  if (image instanceof HTMLVideoElement) return image.videoHeight;
  return image.height;
}
