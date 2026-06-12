# face-hair-cutout-js

Browser ESM utility for creating transparent PNG head cutouts.

The final alpha contour comes from a MediaPipe `ImageSegmenter` category mask.
The `FaceLandmarker` is used only to estimate the chin and face height, then
remove pixels below the chin with a small feathered margin. Hair is not cropped
with a face bounding box.

## Install

```sh
npm install
```

## Usage

```js
import { createCutoutTasks, cutoutFaceHair } from "./src/index.js";

const tasks = await createCutoutTasks({
  segmenterModelPath:
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
});

const result = await cutoutFaceHair(imageElement, tasks, {
  keepCategoryIndexes: [1, 3], // MediaPipe Multi-class Selfie: hair + face-skin
  accessoryCategoryIndexes: [5], // keep glasses/accessories near the face only
  maxWidth: 768,
  maxHeight: 768,
  chinMarginRatio: 0.06,
  bottomFeatherRatio: 0.025
});

document.body.append(result.canvas);
```

By default, this uses MediaPipe's Multi-class Selfie Segmenter. Its category ids
are `0 background`, `1 hair`, `2 body-skin`, `3 face-skin`, `4 clothes`, and
`5 accessories`, so the default cutout keeps `1` and `3`, plus category `5`
inside an expanded face region for glasses.

## Demo

```sh
npm run dev
```

The demo is prefilled with the Multi-class Selfie Segmenter URL. You can replace
it with another Image Segmenter model URL if you also update
`keepCategoryIndexes` in `main.js`.

The model list is documented on Google's MediaPipe Image Segmenter page:
https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter#models

## Using in another website

With a bundler such as Vite, import from `src/index.js` and install the package
dependencies:

```js
import { createCutoutTasks, cutoutFaceHair } from "./src/index.js";
```

For a plain static site without a bundler, keep an import map for
`@mediapipe/tasks-vision`, like the one in `index.html`.
