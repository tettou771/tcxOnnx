#pragma once

// =============================================================================
// tcxOnnx - a thin, model-agnostic wrapper around ONNX Runtime.
//
// Design goals:
//   - ONE addon owns ONNX Runtime. Other addons / apps that need inference link
//     tcxOnnx instead of vendoring their own copy of ORT (which would clash at
//     link time on duplicate symbols and on the onnxruntime headers).
//   - No ONNX Runtime types leak into this header. Inputs/outputs are plain
//     value tensors (onnx::Tensor). Consumers never include onnxruntime headers
//     for the common case; an escape hatch (nativeSession()) is provided for
//     the rare case that needs the raw Ort::Session.
//   - One general run()/kick() that handles any number of named inputs/outputs
//     and mixed element types — so a single API serves face-landmark
//     multi-output detectors, image embedders, text encoders, etc.
//   - Identical contract on native and web. Inference on the web backend
//     (onnxruntime-web) is asynchronous, so the *honest* primitive is the
//     non-blocking kick()/hasResult()/takeResult() trio, which behaves the same
//     on both platforms. run() is a native-synchronous convenience (see below).
// =============================================================================

#include <TrussC.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

// Addons live in the `tcx` namespace; this one nests under `tcx::onnx`
// (mirroring tcx::box2d) so the type names stay short: onnx::Model, onnx::Tensor.
namespace tcx::onnx {

// -----------------------------------------------------------------------------
// Tensor - a plain CPU tensor (data + shape + dtype). No Ort types.
// -----------------------------------------------------------------------------
struct Tensor {
    enum class Type { Float32, Int64, Int32, UInt8, Other };

    Type type = Type::Float32;
    std::vector<int64_t> shape;     // e.g. {1, 3, 192, 192}
    std::vector<uint8_t> bytes;     // raw element bytes, interpreted by `type`

    bool empty() const { return bytes.empty(); }
    // Number of elements = product(shape). Returns 0 if any dim is negative.
    size_t count() const;
    // Size in bytes of one element for `type` (0 for Other/unknown).
    size_t elementSize() const;

    // Builders (take ownership of the data, reinterpret as bytes).
    static Tensor f32(std::vector<float> data, std::vector<int64_t> shape);
    static Tensor i64(std::vector<int64_t> data, std::vector<int64_t> shape);
    static Tensor i32(std::vector<int32_t> data, std::vector<int64_t> shape);
    static Tensor u8(std::vector<uint8_t> data, std::vector<int64_t> shape);

    // Typed copies out. Empty if `type` does not match.
    std::vector<float>   asFloat() const;
    std::vector<int64_t> asInt64() const;
    std::vector<int32_t> asInt32() const;
    std::vector<uint8_t> asUInt8() const;
};

// -----------------------------------------------------------------------------
// Result - the named outputs of one inference. A thin map<name, Tensor> wrapper
// with ergonomic access, so callers never do the find()/end() dance.
// -----------------------------------------------------------------------------
class Result {
public:
    Result() = default;
    explicit Result(std::map<std::string, Tensor> outputs) : outputs_(std::move(outputs)) {}

    bool empty() const { return outputs_.empty(); }
    size_t size() const { return outputs_.size(); }
    bool has(const std::string& name) const { return outputs_.count(name) != 0; }
    // The named output, or a shared empty Tensor if absent (so `.get(...).asFloat()`
    // is always safe — it just yields an empty vector when the output is missing).
    const Tensor& get(const std::string& name) const;
    std::vector<std::string> names() const;

    // Raw access / iteration.
    const std::map<std::string, Tensor>& map() const { return outputs_; }
    auto begin() const { return outputs_.begin(); }
    auto end() const { return outputs_.end(); }

private:
    std::map<std::string, Tensor> outputs_;
};

// -----------------------------------------------------------------------------
// Model - one loaded model / inference session.
// -----------------------------------------------------------------------------
class Model {
public:
    struct Options {
        // "Auto" picks the best available EP per platform (CoreML on macOS,
        // CUDA on Linux when present) and silently falls back to CPU.
        // Force one with "CoreML" / "CUDA" / "CPU".
        // NOTE (web): ort-web ignores these — it runs the wasm/WebGPU backend it
        // was configured with. executionProvider / numThreads / cacheDir are
        // no-ops on the Emscripten build.
        std::string executionProvider = "Auto";
        int numThreads = 2;
        // CoreML compiled-model cache dir. Empty -> ~/.tcxOnnx_cache. Setting
        // this avoids ORT recreating a temp dir on every launch.
        std::string cacheDir = "";
        bool verbose = false;
    };

    Model();
    ~Model();
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Load a .onnx model. Returns false on failure (logs the reason).
    // Call on the main thread (the shared Ort::Env init is main-thread only).
    // NOTE (web): loading is asynchronous — load() returns immediately and
    // isLoaded() flips to true once ort-web + the session are actually ready.
    bool load(const std::string& modelPath, const Options& opts);
    bool load(const std::string& modelPath) { return load(modelPath, Options{}); }
    bool isLoaded() const;
    void unload();

    // ---- Non-blocking inference (the honest, cross-platform primitive) -------
    // kick() schedules one inference over `namedInputs` (native: runs it now into
    // an internal result slot; web: fires the ort-web promise). Then poll
    // hasResult() and consume with takeResult(). The typical realtime loop:
    //
    //     model.kick(inputs);
    //     if (model.hasResult()) { auto r = model.takeResult(); use(r.get("out")); }
    //
    // outputNames selects which outputs to fetch; empty -> all outputs.
    void kick(const std::map<std::string, Tensor>& namedInputs,
              const std::vector<std::string>& outputNames = {});
    // Single-input convenience: uses the model's sole input name.
    void kick(const Tensor& singleInput);
    bool hasResult() const;
    Result takeResult();

    // ---- Synchronous convenience ---------------------------------------------
    // NATIVE: blocks and returns this call's result.
    // WEB: cannot block — kicks the run and returns the most-recent completed
    // result (empty until the first one lands), logging a one-time warning. For
    // realtime web code use kick()/hasResult()/takeResult() instead.
    Result run(const std::map<std::string, Tensor>& namedInputs,
               const std::vector<std::string>& outputNames = {});
    // Single-input convenience: uses the model's sole input name.
    Result run(const Tensor& singleInput);

    // ---- Metadata ------------------------------------------------------------
    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;     // dynamic dims reported as -1
        Tensor::Type type = Tensor::Type::Other;
    };
    std::vector<TensorInfo> inputInfo() const;
    std::vector<TensorInfo> outputInfo() const;
    std::vector<std::string> inputNames() const;
    std::vector<std::string> outputNames() const;
    void printModelInfo() const;

    // Escape hatch: raw Ort::Session* (cast it yourself). nullptr if unloaded or
    // on the web backend.
    void* nativeSession();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tcx::onnx
