# tcxOnnx

ONNX Runtime wrapper for TrussC. Load a `.onnx` model and run inference with a
single, model-agnostic API. No OpenCV dependency — you feed it plain tensors.

This is meant to be the **one** addon that owns ONNX Runtime. Other addons or
apps that need inference should depend on `tcxOnnx` rather than vendoring their
own ORT (two copies would clash on duplicate symbols and on the onnxruntime
headers).

Everything lives in the `tcx::onnx` namespace: `onnx::Model`, `onnx::Tensor`,
`onnx::Result`.

## Setup

The prebuilt ONNX Runtime binaries are not committed. Fetch them once:

```bash
./scripts/fetch_onnxruntime.sh          # downloads ORT for your platform into lib/
```

Then add the addon to your project's `addons.make`:

```
tcxOnnx
```

## Usage

```cpp
#include <tcxOnnx.h>
using namespace tcx;

onnx::Model model;
model.load("data/models/blazeface.onnx");   // CoreML on macOS, CPU elsewhere
model.printModelInfo();                       // dump input/output names + shapes

// General inference: named inputs -> named outputs.
onnx::Result r = model.run({
    {"image", onnx::Tensor::f32(std::move(pixels), {1, 3, 128, 128})},
});
std::vector<float> boxes = r.get("boxes").asFloat();   // empty if "boxes" is absent

// Single-input convenience: uses the model's sole input name.
onnx::Result r2 = model.run(onnx::Tensor::f32(std::move(img), {1, 1, 28, 28}));
```

`run()` handles any number of named inputs/outputs and mixed element types
(float32 / int64 / int32 / uint8), so one call serves multi-output detectors,
image embedders, and text encoders alike. `Result::get(name)` returns a shared
empty `Tensor` when an output is absent, so `.get(...).asFloat()` is always safe.

### Non-blocking inference (realtime + web)

ONNX Runtime on the web backend (`onnxruntime-web`) is **asynchronous**. The
honest, cross-platform primitive is therefore the non-blocking trio — identical
behavior on native and web:

```cpp
model.kick(inputs);                          // schedule one inference
if (model.hasResult()) {                     // poll
    onnx::Result r = model.takeResult();     // consume the completed result
    use(r.get("out").asFloat());
}
```

- **Native**: `kick()` runs synchronously into a result slot; `hasResult()` is
  true immediately. `run()` is a blocking convenience that returns the result.
- **Web**: `kick()` fires the ort-web promise and returns at once; `hasResult()`
  flips true a few frames later. `run()` cannot block — it returns the
  most-recent completed result (lagging a few frames) and warns once. Use
  `kick()`/`hasResult()`/`takeResult()` for realtime web code.

For a two-stage pipeline (detector → cropper → landmarks), drive it as a small
state machine over `kick`/`hasResult`/`takeResult` — see kandecrash's
`FaceTracker` for a worked example.

### Notes

- Call `load()` on the main thread (the shared `Ort::Env` init is main-thread
  only). On web, loading is asynchronous: `load()` returns immediately and
  `isLoaded()` flips true once ort-web + the session are ready.
- `Options` (`executionProvider` / `numThreads` / `cacheDir`) are honored on
  native; on web ort-web ignores them (it runs its own wasm/WebGPU backend).
- Post-processing (anchor decode, NMS, L2-normalize, …) is left to the caller —
  this addon returns the raw model outputs.
- Escape hatch: `model.nativeSession()` returns the raw `Ort::Session*` if you
  need the full ORT API (native only; `nullptr` on web).

## Example

`example-basic` is an MNIST handwritten-digit demo: draw a digit with the mouse
and it runs the bundled 26 KB model every frame the canvas changes. It exercises
`load()` → single-input `run(Tensor)` → `Result::get()` → `Tensor::asFloat()`.

```bash
cd example-basic
cmake --preset macos && cmake --build --preset macos
./bin/example-basic.app/Contents/MacOS/example-basic
```

## Platforms

| Platform | Backend | Status |
|----------|---------|--------|
| macOS (arm64) | CoreML EP + CPU | ✅ |
| Linux (x64) | CUDA EP (if present) + CPU | ✅ |
| Windows (x64) | CPU | ✅ |
| Web (wasm) | onnxruntime-web (CDN, non-blocking bridge) | ✅ |

## License

MIT (this addon). ONNX Runtime is MIT-licensed by Microsoft; the bundled MNIST
example model is Apache-2.0 (ONNX Model Zoo) — see `LICENSES.md`.
