#pragma once

// =============================================================================
// tcxOnnx - a thin, model-agnostic wrapper around ONNX Runtime.
//
// Design goals:
//   - ONE addon owns ONNX Runtime. Other addons / apps that need inference link
//     tcxOnnx instead of vendoring their own copy of ORT (which would clash at
//     link time on duplicate symbols and on the onnxruntime headers).
//   - No ONNX Runtime types leak into this header. Inputs/outputs are plain
//     value tensors (OnnxTensor). Consumers never include onnxruntime headers
//     for the common case; an escape hatch (nativeSession()) is provided for
//     the rare case that needs the raw Ort::Session.
//   - One general run() that handles any number of named inputs / outputs and
//     mixed element types — so a single API serves face-landmark multi-output
//     detectors, image embedders, text encoders, etc.
// =============================================================================

#include <TrussC.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

// Addons live in the `tcx` namespace (framework core uses `tc`).
namespace tcx {

// -----------------------------------------------------------------------------
// OnnxTensor - a plain CPU tensor (data + shape + dtype). No Ort types.
// -----------------------------------------------------------------------------
struct OnnxTensor {
    enum class Type { Float32, Int64, Int32, UInt8, Other };

    Type type = Type::Float32;
    std::vector<int64_t> shape;     // e.g. {1, 3, 192, 192}
    std::vector<uint8_t> bytes;     // raw element bytes, interpreted by `type`

    // Number of elements = product(shape). Returns 0 if any dim is negative.
    size_t count() const;
    // Size in bytes of one element for `type` (0 for Other/unknown).
    size_t elementSize() const;

    // Builders (take ownership of the data, reinterpret as bytes).
    static OnnxTensor f32(std::vector<float> data, std::vector<int64_t> shape);
    static OnnxTensor i64(std::vector<int64_t> data, std::vector<int64_t> shape);
    static OnnxTensor i32(std::vector<int32_t> data, std::vector<int64_t> shape);
    static OnnxTensor u8(std::vector<uint8_t> data, std::vector<int64_t> shape);

    // Typed copies out. Empty if `type` does not match.
    std::vector<float>   asFloat() const;
    std::vector<int64_t> asInt64() const;
    std::vector<int32_t> asInt32() const;
};

// -----------------------------------------------------------------------------
// OnnxModel - one loaded model / inference session.
// -----------------------------------------------------------------------------
class OnnxModel {
public:
    struct Options {
        // "Auto" picks the best available EP per platform (CoreML on macOS,
        // CUDA on Linux when present) and silently falls back to CPU.
        // Force one with "CoreML" / "CUDA" / "CPU".
        std::string executionProvider = "Auto";
        int numThreads = 2;
        // CoreML compiled-model cache dir. Empty -> ~/.tcxOnnx_cache. Setting
        // this avoids ORT recreating a temp dir on every launch.
        std::string cacheDir = "";
        bool verbose = false;
    };

    OnnxModel();
    ~OnnxModel();
    OnnxModel(OnnxModel&&) noexcept;
    OnnxModel& operator=(OnnxModel&&) noexcept;
    OnnxModel(const OnnxModel&) = delete;
    OnnxModel& operator=(const OnnxModel&) = delete;

    // Load a .onnx model. Returns false on failure (logs the reason).
    // Call on the main thread (the shared Ort::Env init is main-thread only).
    bool load(const std::string& modelPath, const Options& opts);
    bool load(const std::string& modelPath) { return load(modelPath, Options{}); }
    bool isLoaded() const;
    void unload();

    // General inference.
    //   namedInputs : keyed by the model's input tensor names.
    //   outputNames : which outputs to fetch; empty -> all model outputs.
    // Returns outputs keyed by name (empty map on error).
    std::map<std::string, OnnxTensor> run(
        const std::map<std::string, OnnxTensor>& namedInputs,
        const std::vector<std::string>& outputNames = {});

    // Model metadata (dynamic dims are reported as -1).
    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        OnnxTensor::Type type = OnnxTensor::Type::Other;
    };
    std::vector<TensorInfo> inputInfo() const;
    std::vector<TensorInfo> outputInfo() const;
    void printModelInfo() const;

    // Escape hatch: raw Ort::Session* (cast it yourself). nullptr if unloaded.
    void* nativeSession();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tcx
