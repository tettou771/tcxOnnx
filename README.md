# tcxOnnx

> ⚠️ **Work in progress.** This addon is under active development and its API
> may change without notice. Not yet released — use at your own risk until a
> tagged `v0.1.0`.

ONNX Runtime wrapper for TrussC. Load a `.onnx` model and run inference with a
single, model-agnostic API. No OpenCV dependency — you feed it plain tensors.

This is meant to be the **one** addon that owns ONNX Runtime. Other addons or
apps that need inference should depend on `tcxOnnx` rather than vendoring their
own ORT (two copies would clash on duplicate symbols and on the onnxruntime
headers).

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

OnnxModel model;
model.load("data/models/blazeface.onnx");   // CoreML on macOS, CPU elsewhere
model.printModelInfo();                       // dump input/output names + shapes

// General inference: named inputs -> named outputs.
auto inputs = std::map<std::string, OnnxTensor>{
    {"input", OnnxTensor::f32(std::move(pixels), {1, 3, 128, 128})}
};
auto outputs = model.run(inputs);             // empty outputNames -> all outputs
std::vector<float> boxes = outputs["boxes"].asFloat();
```

`OnnxModel::run()` handles any number of named inputs/outputs and mixed element
types (float32 / int64 / int32 / uint8), so one call serves multi-output
detectors, image embedders, and text encoders alike.

### Notes

- Call `load()` on the main thread (the shared `Ort::Env` init is main-thread
  only). `run()` is safe to call from a worker thread once loaded.
- Post-processing (anchor decode, NMS, L2-normalize, …) is left to the caller —
  this addon returns the raw model outputs.
- Escape hatch: `model.nativeSession()` returns the raw `Ort::Session*` if you
  need the full ORT API (include the onnxruntime headers yourself).

## Platforms

| Platform | Backend | Status |
|----------|---------|--------|
| macOS (arm64) | CoreML EP + CPU | ✅ |
| Linux (x64) | CUDA EP (if present) + CPU | ✅ |
| Windows (x64) | CPU | ✅ |
| Web (wasm) | — | planned (M5: `--build_wasm_static_lib` or TFLite) |

## License

MIT (this addon). ONNX Runtime is MIT-licensed by Microsoft — see `LICENSES.md`.
