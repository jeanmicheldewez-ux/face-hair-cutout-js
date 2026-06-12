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
 * @param {number} [options.chinMarginRatio]
 * @param {number} [options.bottomFeatherRatio]
 * @param {number} [options.maskFeatherPx]
 * @returns {Promise<{canvas: HTMLCanvasElement, pngBlob: Blob, pngDataUrl: string, bounds: {x: number, y: number, width: number, height: number}}>}
 */
export async function cutoutFaceHair(image, tasks, options = {}) {
  const {
    keepCategoryIndexes = [1, 3],
    accessoryCategoryIndexes = [5],
    accessoryFaceExpansionRatio = 0.18,
    maxWidth = 1024,
    maxHeight = 1024,
    chinMarginRatio = 0.06,
    bottomFeatherRatio = 0.02,
    maskFeatherPx = 1
  } = options;

  const source = drawToCanvas(image);
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
  const imageData = get2d(source).getImageData(0, 0, width, height);

  for (let i = 0; i < refinedAlpha.length; i += 1) {
    imageData.data[i * 4 + 3] = refinedAlpha[i];
  }

  const alphaCanvas = makeCanvas(width, height);
  get2d(alphaCanvas).putImageData(imageData, 0, 0);

  const bounds = getAlphaBounds(refinedAlpha, width, height);
  if (!bounds) {
    throw new Error("The final mask is empty. Check segmenter category indexes.");
  }

  const cropped = cropAndResize(alphaCanvas, bounds, maxWidth, maxHeight);
  const pngBlob = await canvasToBlob(cropped);

  return {
    canvas: cropped,
    pngBlob,
    pngDataUrl: cropped.toDataURL("image/png"),
    bounds
  };
}

/**
 * @param {CutoutImageSource} image
 * @returns {HTMLCanvasElement}
 */
function drawToCanvas(image) {
  const width = Math.round(getSourceWidth(image));
  const height = Math.round(getSourceHeight(image));
  const canvas = makeCanvas(width, height);
  get2d(canvas).drawImage(image, 0, 0, width, height);
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

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      const maskX = Math.min(maskWidth - 1, Math.floor((x / width) * maskWidth));
      const maskY = Math.min(maskHeight - 1, Math.floor((y / height) * maskHeight));
      const category = mask[maskY * maskWidth + maskX];
      const keepMainCategory = keepCategoryIndexes.has(category);
      const keepFaceAccessory =
        accessoryCategoryIndexes.has(category) && isInsideRegion(x, y, accessoryRegion);
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
 * @param {HTMLCanvasElement} source
 * @param {{x: number, y: number, width: number, height: number}} bounds
 * @param {number} maxWidth
 * @param {number} maxHeight
 * @returns {HTMLCanvasElement}
 */
function cropAndResize(source, bounds, maxWidth, maxHeight) {
  const scale = Math.min(1, maxWidth / bounds.width, maxHeight / bounds.height);
  const output = makeCanvas(
    Math.max(1, Math.round(bounds.width * scale)),
    Math.max(1, Math.round(bounds.height * scale))
  );
  get2d(output).drawImage(
    source,
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height,
    0,
    0,
    output.width,
    output.height
  );
  return output;
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
