// =============================================================================
// tcxOnnx.cpp - ONNX Runtime is included ONLY here, behind a PIMPL.
// =============================================================================

#include "tcxOnnx.h"

#include <onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#elif defined(__linux__) && __has_include(<cuda_provider_factory.h>)
#define TCXONNX_HAS_CUDA 1
#include <cuda_provider_factory.h>
#endif

#include <filesystem>
#include <cstdlib>

using namespace std;
using namespace tc;

namespace tcx {

// -----------------------------------------------------------------------------
// Type mapping helpers
// -----------------------------------------------------------------------------
static OnnxTensor::Type fromOrt(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return OnnxTensor::Type::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return OnnxTensor::Type::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return OnnxTensor::Type::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return OnnxTensor::Type::UInt8;
        default: return OnnxTensor::Type::Other;
    }
}

static const char* typeName(OnnxTensor::Type t) {
    switch (t) {
        case OnnxTensor::Type::Float32: return "float32";
        case OnnxTensor::Type::Int64:   return "int64";
        case OnnxTensor::Type::Int32:   return "int32";
        case OnnxTensor::Type::UInt8:   return "uint8";
        default:                        return "other";
    }
}

// -----------------------------------------------------------------------------
// OnnxTensor
// -----------------------------------------------------------------------------
size_t OnnxTensor::elementSize() const {
    switch (type) {
        case Type::Float32: return 4;
        case Type::Int64:   return 8;
        case Type::Int32:   return 4;
        case Type::UInt8:   return 1;
        default:            return 0;
    }
}

size_t OnnxTensor::count() const {
    size_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) return 0;
        n *= static_cast<size_t>(d);
    }
    return shape.empty() ? 0 : n;
}

template <typename T>
static OnnxTensor makeTensor(OnnxTensor::Type type, vector<T> data, vector<int64_t> shape) {
    OnnxTensor t;
    t.type = type;
    t.shape = std::move(shape);
    t.bytes.resize(data.size() * sizeof(T));
    if (!data.empty()) memcpy(t.bytes.data(), data.data(), t.bytes.size());
    return t;
}

OnnxTensor OnnxTensor::f32(vector<float> d, vector<int64_t> s)   { return makeTensor(Type::Float32, std::move(d), std::move(s)); }
OnnxTensor OnnxTensor::i64(vector<int64_t> d, vector<int64_t> s) { return makeTensor(Type::Int64,   std::move(d), std::move(s)); }
OnnxTensor OnnxTensor::i32(vector<int32_t> d, vector<int64_t> s) { return makeTensor(Type::Int32,   std::move(d), std::move(s)); }
OnnxTensor OnnxTensor::u8(vector<uint8_t> d, vector<int64_t> s)  { return makeTensor(Type::UInt8,   std::move(d), std::move(s)); }

template <typename T>
static vector<T> typedCopy(const OnnxTensor& t, OnnxTensor::Type expect) {
    if (t.type != expect || t.bytes.empty()) return {};
    vector<T> out(t.bytes.size() / sizeof(T));
    memcpy(out.data(), t.bytes.data(), out.size() * sizeof(T));
    return out;
}

vector<float>   OnnxTensor::asFloat() const { return typedCopy<float>(*this, Type::Float32); }
vector<int64_t> OnnxTensor::asInt64() const { return typedCopy<int64_t>(*this, Type::Int64); }
vector<int32_t> OnnxTensor::asInt32() const { return typedCopy<int32_t>(*this, Type::Int32); }

// -----------------------------------------------------------------------------
// Shared environment & cache dir
// -----------------------------------------------------------------------------
// One Ort::Env per process (recommended). Main-thread init.
static Ort::Env& sharedEnv() {
    // ERROR level: post-processing-merged models legitimately emit dynamic
    // output shapes that differ from the static graph shape, which would spam
    // per-frame "VerifyOutputSizes" warnings at WARNING level.
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "tcxOnnx");
    return env;
}

static string defaultCacheDir() {
    const char* home = getenv("HOME");
    string base = home ? home : ".";
    string path = base + "/.tcxOnnx_cache";
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return path;
}

// -----------------------------------------------------------------------------
// PIMPL
// -----------------------------------------------------------------------------
struct OnnxModel::Impl {
    unique_ptr<Ort::Session> session;
    // Cache input/output names so we don't re-allocate them every run.
    vector<string> inputNames;
    vector<string> outputNames;
};

OnnxModel::OnnxModel() : impl_(make_unique<Impl>()) {}
OnnxModel::~OnnxModel() = default;
OnnxModel::OnnxModel(OnnxModel&&) noexcept = default;
OnnxModel& OnnxModel::operator=(OnnxModel&&) noexcept = default;

static void configureEP(Ort::SessionOptions& opts, const OnnxModel::Options& o, const string& cacheDir) {
    const string& ep = o.executionProvider;
#if defined(__APPLE__)
    if (ep == "Auto" || ep == "CoreML") {
        try {
            opts.AppendExecutionProvider("CoreML", {
                {kCoremlProviderOption_ModelCacheDirectory, cacheDir}
            });
            if (o.verbose) logNotice() << "[tcxOnnx] CoreML EP (cache: " << cacheDir << ")";
            return;
        } catch (const std::exception& e) {
            logWarning() << "[tcxOnnx] CoreML EP unavailable, falling back to CPU: " << e.what();
        }
    }
#elif defined(TCXONNX_HAS_CUDA)
    if (ep == "Auto" || ep == "CUDA") {
        try {
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            opts.AppendExecutionProviderCUDA(cuda);
            if (o.verbose) logNotice() << "[tcxOnnx] CUDA EP";
            return;
        } catch (const std::exception& e) {
            logWarning() << "[tcxOnnx] CUDA EP unavailable, falling back to CPU: " << e.what();
        }
    }
#else
    (void)cacheDir;
#endif
    if (o.verbose) logNotice() << "[tcxOnnx] CPU EP";
}

bool OnnxModel::load(const string& modelPath, const Options& opts) {
    try {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(opts.numThreads);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        string cacheDir = opts.cacheDir.empty() ? defaultCacheDir() : opts.cacheDir;
        configureEP(so, opts, cacheDir);

        impl_->session = make_unique<Ort::Session>(sharedEnv(), modelPath.c_str(), so);

        // Cache I/O names once.
        Ort::AllocatorWithDefaultOptions alloc;
        impl_->inputNames.clear();
        impl_->outputNames.clear();
        size_t ni = impl_->session->GetInputCount();
        for (size_t i = 0; i < ni; i++)
            impl_->inputNames.push_back(impl_->session->GetInputNameAllocated(i, alloc).get());
        size_t no = impl_->session->GetOutputCount();
        for (size_t i = 0; i < no; i++)
            impl_->outputNames.push_back(impl_->session->GetOutputNameAllocated(i, alloc).get());

        if (opts.verbose) logNotice() << "[tcxOnnx] loaded: " << modelPath;
        return true;
    } catch (const Ort::Exception& e) {
        logError() << "[tcxOnnx] failed to load '" << modelPath << "': " << e.what();
        impl_->session.reset();
        return false;
    }
}

bool OnnxModel::isLoaded() const { return impl_ && impl_->session != nullptr; }

void OnnxModel::unload() {
    if (impl_) {
        impl_->session.reset();
        impl_->inputNames.clear();
        impl_->outputNames.clear();
    }
}

// Build an Ort::Value that references (does not copy) the tensor's bytes.
// The source OnnxTensor must outlive the Run() call.
static Ort::Value makeOrtValue(const Ort::MemoryInfo& mem, const OnnxTensor& t) {
    void* data = const_cast<uint8_t*>(t.bytes.data());
    size_t n = t.count();
    switch (t.type) {
        case OnnxTensor::Type::Float32:
            return Ort::Value::CreateTensor<float>(mem, reinterpret_cast<float*>(data), n, t.shape.data(), t.shape.size());
        case OnnxTensor::Type::Int64:
            return Ort::Value::CreateTensor<int64_t>(mem, reinterpret_cast<int64_t*>(data), n, t.shape.data(), t.shape.size());
        case OnnxTensor::Type::Int32:
            return Ort::Value::CreateTensor<int32_t>(mem, reinterpret_cast<int32_t*>(data), n, t.shape.data(), t.shape.size());
        case OnnxTensor::Type::UInt8:
            return Ort::Value::CreateTensor<uint8_t>(mem, reinterpret_cast<uint8_t*>(data), n, t.shape.data(), t.shape.size());
        default:
            throw std::runtime_error("tcxOnnx: unsupported input tensor type");
    }
}

static OnnxTensor toTensor(Ort::Value& v) {
    OnnxTensor out;
    auto info = v.GetTensorTypeAndShapeInfo();
    out.shape = info.GetShape();
    out.type = fromOrt(info.GetElementType());
    size_t n = info.GetElementCount();
    size_t es = out.elementSize();
    if (es == 0) {
        // Unknown element type: copy raw bytes by best effort (float-sized fallback).
        es = 4;
    }
    const void* src = v.GetTensorRawData();
    out.bytes.resize(n * es);
    if (n) memcpy(out.bytes.data(), src, out.bytes.size());
    return out;
}

map<string, OnnxTensor> OnnxModel::run(
    const map<string, OnnxTensor>& namedInputs,
    const vector<string>& outputNames) {
    map<string, OnnxTensor> result;
    if (!isLoaded()) {
        logError() << "[tcxOnnx] run() called on an unloaded model";
        return result;
    }
    try {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        vector<const char*> inNames;
        vector<Ort::Value> inValues;
        inNames.reserve(namedInputs.size());
        inValues.reserve(namedInputs.size());
        for (const auto& kv : namedInputs) {
            inNames.push_back(kv.first.c_str());
            inValues.push_back(makeOrtValue(mem, kv.second));
        }

        // Resolve which outputs to fetch.
        const vector<string>& outStore = outputNames.empty() ? impl_->outputNames : outputNames;
        vector<const char*> outNames;
        outNames.reserve(outStore.size());
        for (const auto& s : outStore) outNames.push_back(s.c_str());

        auto outputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            inNames.data(), inValues.data(), inValues.size(),
            outNames.data(), outNames.size());

        for (size_t i = 0; i < outputs.size() && i < outStore.size(); i++) {
            if (outputs[i].IsTensor())
                result[outStore[i]] = toTensor(outputs[i]);
        }
    } catch (const Ort::Exception& e) {
        logError() << "[tcxOnnx] run failed: " << e.what();
    } catch (const std::exception& e) {
        logError() << "[tcxOnnx] run error: " << e.what();
    }
    return result;
}

static vector<OnnxModel::TensorInfo> collectInfo(Ort::Session* s, bool inputs) {
    vector<OnnxModel::TensorInfo> infos;
    if (!s) return infos;
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n = inputs ? s->GetInputCount() : s->GetOutputCount();
    for (size_t i = 0; i < n; i++) {
        OnnxModel::TensorInfo ti;
        ti.name = inputs ? s->GetInputNameAllocated(i, alloc).get()
                         : s->GetOutputNameAllocated(i, alloc).get();
        auto typeInfo = inputs ? s->GetInputTypeInfo(i) : s->GetOutputTypeInfo(i);
        auto tinfo = typeInfo.GetTensorTypeAndShapeInfo();
        ti.shape = tinfo.GetShape();
        ti.type = fromOrt(tinfo.GetElementType());
        infos.push_back(std::move(ti));
    }
    return infos;
}

vector<OnnxModel::TensorInfo> OnnxModel::inputInfo() const {
    return collectInfo(impl_ ? impl_->session.get() : nullptr, true);
}
vector<OnnxModel::TensorInfo> OnnxModel::outputInfo() const {
    return collectInfo(impl_ ? impl_->session.get() : nullptr, false);
}

static string shapeStr(const vector<int64_t>& shape) {
    string s;
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) s += ", ";
        s += to_string(shape[i]);
    }
    return s;
}

void OnnxModel::printModelInfo() const {
    if (!isLoaded()) { logNotice() << "[tcxOnnx] (no model loaded)"; return; }
    for (const auto& ti : inputInfo())
        logNotice() << "[tcxOnnx] input  " << ti.name << " " << typeName(ti.type) << " [" << shapeStr(ti.shape) << "]";
    for (const auto& ti : outputInfo())
        logNotice() << "[tcxOnnx] output " << ti.name << " " << typeName(ti.type) << " [" << shapeStr(ti.shape) << "]";
}

void* OnnxModel::nativeSession() {
    return (impl_ && impl_->session) ? static_cast<void*>(impl_->session.get()) : nullptr;
}

} // namespace tcx
