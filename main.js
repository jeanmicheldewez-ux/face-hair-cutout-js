import { createCutoutTasks, cutoutFaceHair } from "./src/index.js";

const DEFAULT_SEGMENTER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";

/** @type {HTMLInputElement} */
const fileInput = getElement("#file", HTMLInputElement);
/** @type {HTMLInputElement} */
const modelInput = getElement("#model", HTMLInputElement);
/** @type {HTMLButtonElement} */
const runButton = getElement("#run", HTMLButtonElement);
/** @type {HTMLCanvasElement} */
const sourceCanvas = getElement("#source", HTMLCanvasElement);
/** @type {HTMLCanvasElement} */
const resultCanvas = getElement("#result", HTMLCanvasElement);
/** @type {HTMLAnchorElement} */
const downloadLink = getElement("#download", HTMLAnchorElement);
/** @type {HTMLParagraphElement} */
const statusText = getElement("#status", HTMLParagraphElement);

/** @type {ImageBitmap | null} */
let sourceImage = null;
/** @type {Promise<any> | null} */
let tasksPromise = null;
let currentModelPath = "";

modelInput.value = DEFAULT_SEGMENTER_MODEL;

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  sourceImage = await createImageBitmap(file);
  setSourceImage(sourceImage, file.name);
});

runButton.addEventListener("click", async () => {
  if (!sourceImage) return;

  runButton.disabled = true;
  statusText.textContent = "Loading models...";

  try {
    const segmenterModelPath = modelInput.value.trim() || DEFAULT_SEGMENTER_MODEL;
    if (segmenterModelPath !== currentModelPath) {
      tasksPromise = createCutoutTasks({ segmenterModelPath });
      currentModelPath = segmenterModelPath;
    }
    const tasks = await tasksPromise;
    statusText.textContent = "Segmenting...";

    const result = await cutoutFaceHair(sourceImage, tasks, {
      maxWidth: 480,
      maxHeight: 640,
      topPaddingRatio: 0.25,
      sidePaddingRatio: 0.14,
      bottomPaddingRatio: 0.06,
      inputScaleRatio: 0.75,
      outputFit: "contain",
      outputTrimTopRatio: 0.07,
      outputTrimBottomRatio: 0.02,
      debugCrop: false,
      keepCategoryIndexes: [1, 3],
      accessoryCategoryIndexes: [5],
      accessoryFaceExpansionRatio: 0.18,
      chinMarginRatio: 0.06,
      bottomFeatherRatio: 0.025
    });

    resultCanvas.width = result.canvas.width;
    resultCanvas.height = result.canvas.height;
    get2d(resultCanvas).drawImage(result.canvas, 0, 0);
    downloadLink.href = result.pngDataUrl;
    downloadLink.hidden = false;
    console.log("face-hair-cutout result", {
      rawMaskBox: result.rawMaskBox,
      paddedCropBox: result.paddedCropBox,
      requestedCropBox: result.requestedCropBox,
      cropCanvasSize: result.cropCanvasSize,
      outputSize: result.outputSize,
      resize: result.resize,
      outputTrim: result.outputTrim
    });
    statusText.textContent = `Done: ${result.canvas.width} x ${result.canvas.height}`;
  } catch (error) {
    console.error(error);
    statusText.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    runButton.disabled = false;
  }
});

/**
 * @param {ImageBitmap} image
 * @param {string} label
 */
function setSourceImage(image, label) {
  drawSource(image);
  clearCanvas(resultCanvas);
  runButton.disabled = false;
  downloadLink.hidden = true;
  statusText.textContent = `Ready: ${label}`;
}

/**
 * @param {ImageBitmap} image
 */
function drawSource(image) {
  const scale = Math.min(1, 720 / image.width, 720 / image.height);
  sourceCanvas.width = Math.round(image.width * scale);
  sourceCanvas.height = Math.round(image.height * scale);
  get2d(sourceCanvas).drawImage(image, 0, 0, sourceCanvas.width, sourceCanvas.height);
}

/**
 * @param {HTMLCanvasElement} canvas
 */
function clearCanvas(canvas) {
  const context = get2d(canvas);
  context.clearRect(0, 0, canvas.width, canvas.height);
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
 * @template {Element} T
 * @param {string} selector
 * @param {new (...args: any[]) => T} constructor
 * @returns {T}
 */
function getElement(selector, constructor) {
  const element = document.querySelector(selector);
  if (!(element instanceof constructor)) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}
