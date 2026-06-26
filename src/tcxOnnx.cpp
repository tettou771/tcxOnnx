// =============================================================================
// tcxOnnx.cpp - ONNX Runtime is included ONLY here, behind a PIMPL.
// =============================================================================

#include "tcxOnnx.h"

#ifndef __EMSCRIPTEN__
#include <onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#elif defined(__linux__) && __has_include(<cuda_provider_factory.h>)
#define TCXONNX_HAS_CUDA 1
#include <cuda_provider_factory.h>
#endif
#include <filesystem>
#else
#include <emscripten.h>
#include <fstream>
#include <algorithm>
#endif

#include <cstdlib>

using namespace std;
using namespace tc;

namespace tcx {

// -----------------------------------------------------------------------------
// Type mapping helpers
// -----------------------------------------------------------------------------
#ifndef __EMSCRIPTEN__
static OnnxTensor::Type fromOrt(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return OnnxTensor::Type::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return OnnxTensor::Type::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return OnnxTensor::Type::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return OnnxTensor::Type::UInt8;
        default: return OnnxTensor::Type::Other;
    }
}
#endif

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

#ifndef __EMSCRIPTEN__
// =============================================================================
// NATIVE backend (ONNX Runtime C++ API)
// =============================================================================

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

#else
// =============================================================================
// WASM backend (onnxruntime-web via an EM_JS bridge).
//
// ASYNCIFY (enabled by trussc_app for Emscripten) lets these synchronous C++
// calls await ort-web's async create()/run(). State lives on Module.__ortx.
// Models are read from the preloaded Emscripten FS (bin/data -> /data).
// =============================================================================

// Inject ort-web from a CDN (once) and await it. Returns 1 when ready.
EM_ASYNC_JS(int, tcxort_init, (), {
    if (!Module.__ortx) Module.__ortx = { ready:false, sessions:[], feeds:{}, outputs:{} };
    if (Module.__ortx.ready) return 1;
    if (typeof ort === 'undefined') {
        await new Promise(function(resolve){
            var s = document.createElement('script');
            s.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.js';
            s.onload = resolve; s.onerror = function(){ console.error('ort-web load failed'); resolve(); };
            document.head.appendChild(s);
        });
    }
    if (typeof ort === 'undefined') return 0;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
    ort.env.wasm.numThreads = 1;   // gh-pages can't set COOP/COEP -> single thread
    Module.__ortx.ready = true;
    return 1;
});

// Create a session from model bytes in the wasm heap. Returns index or -1.
EM_ASYNC_JS(int, tcxort_create, (const unsigned char* data, int len), {
    try {
        var bytes = HEAPU8.slice(data, data + len);
        var s = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
        Module.__ortx.sessions.push(s);
        return Module.__ortx.sessions.length - 1;
    } catch (e) { console.error('ort create failed', e); return -1; }
});

EM_JS(void, tcxort_clear_feeds, (), { Module.__ortx.feeds = {}; });

// Stage one input. type: 0=f32, 1=i64, 2=i32, 3=u8.
EM_JS(void, tcxort_set_input, (const char* namePtr, int type, const void* dataPtr, int count, const int* shapePtr, int ndim), {
    var name = UTF8ToString(namePtr);
    var dims = [];
    for (var i = 0; i < ndim; i++) dims.push(HEAP32[(shapePtr >> 2) + i]);
    var t;
    if (type === 0)      t = new ort.Tensor('float32', HEAPF32.slice(dataPtr >> 2, (dataPtr >> 2) + count), dims);
    else if (type === 1) t = new ort.Tensor('int64', new BigInt64Array(HEAP8.buffer, dataPtr, count).slice(), dims);
    else if (type === 2) t = new ort.Tensor('int32', HEAP32.slice(dataPtr >> 2, (dataPtr >> 2) + count), dims);
    else                 t = new ort.Tensor('uint8', HEAPU8.slice(dataPtr, dataPtr + count), dims);
    Module.__ortx.feeds[name] = t;
});

EM_ASYNC_JS(int, tcxort_run, (int session), {
    try {
        Module.__ortx.outputs = await Module.__ortx.sessions[session].run(Module.__ortx.feeds);
        return 0;
    } catch (e) { console.error('ort run failed', e); return -1; }
});

EM_JS(int, tcxort_output_count, (), { return Object.keys(Module.__ortx.outputs).length; });

EM_JS(int, tcxort_output_name, (int idx, char* buf, int buflen), {
    var k = Object.keys(Module.__ortx.outputs)[idx];
    if (k === undefined) { if (buflen) HEAPU8[buf] = 0; return 0; }
    stringToUTF8(k, buf, buflen);
    return lengthBytesUTF8(k);
});

EM_JS(int, tcxort_output_type, (const char* namePtr), {
    var o = Module.__ortx.outputs[UTF8ToString(namePtr)];
    if (!o) return 4;
    if (o.type === 'float32') return 0;
    if (o.type === 'int64')   return 1;
    if (o.type === 'int32')   return 2;
    if (o.type === 'uint8')   return 3;
    return 4;
});

EM_JS(int, tcxort_output_ndim, (const char* namePtr), {
    var o = Module.__ortx.outputs[UTF8ToString(namePtr)];
    return o ? o.dims.length : 0;
});

EM_JS(void, tcxort_output_shape, (const char* namePtr, int* outShape), {
    var o = Module.__ortx.outputs[UTF8ToString(namePtr)];
    if (!o) return;
    for (var i = 0; i < o.dims.length; i++) HEAP32[(outShape >> 2) + i] = o.dims[i];
});

EM_JS(int, tcxort_output_elems, (const char* namePtr), {
    var o = Module.__ortx.outputs[UTF8ToString(namePtr)];
    return o ? o.data.length : 0;
});

EM_JS(void, tcxort_output_data, (const char* namePtr, void* dst), {
    var o = Module.__ortx.outputs[UTF8ToString(namePtr)];
    if (!o) return;
    var d = o.data;
    if (o.type === 'float32')      HEAPF32.set(d, dst >> 2);
    else if (o.type === 'int32')   HEAP32.set(d, dst >> 2);
    else if (o.type === 'int64')   { var v = new BigInt64Array(HEAP8.buffer, dst, d.length); v.set(d); }
    else                           HEAPU8.set(d, dst);
});

struct OnnxModel::Impl { int session = -1; };

OnnxModel::OnnxModel() : impl_(make_unique<Impl>()) {}
OnnxModel::~OnnxModel() = default;
OnnxModel::OnnxModel(OnnxModel&&) noexcept = default;
OnnxModel& OnnxModel::operator=(OnnxModel&&) noexcept = default;

bool OnnxModel::load(const string& modelPath, const Options& opts) {
    if (!tcxort_init()) { logError() << "[tcxOnnx] ort-web unavailable"; return false; }
    std::ifstream f(modelPath, std::ios::binary);
    if (!f) { logError() << "[tcxOnnx] cannot open " << modelPath; return false; }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    impl_->session = tcxort_create(bytes.data(), (int)bytes.size());
    if (impl_->session < 0) { logError() << "[tcxOnnx] ort-web create failed: " << modelPath; return false; }
    if (opts.verbose) logNotice() << "[tcxOnnx] loaded (ort-web): " << modelPath;
    return true;
}

bool OnnxModel::isLoaded() const { return impl_ && impl_->session >= 0; }
void OnnxModel::unload() { if (impl_) impl_->session = -1; }

static int wasmTypeId(OnnxTensor::Type t) {
    switch (t) { case OnnxTensor::Type::Float32: return 0; case OnnxTensor::Type::Int64: return 1;
                 case OnnxTensor::Type::Int32: return 2; case OnnxTensor::Type::UInt8: return 3; default: return 0; }
}
static OnnxTensor::Type fromWasmTypeId(int t) {
    switch (t) { case 0: return OnnxTensor::Type::Float32; case 1: return OnnxTensor::Type::Int64;
                 case 2: return OnnxTensor::Type::Int32; case 3: return OnnxTensor::Type::UInt8; default: return OnnxTensor::Type::Other; }
}

map<string, OnnxTensor> OnnxModel::run(const map<string, OnnxTensor>& namedInputs, const vector<string>& outputNames) {
    map<string, OnnxTensor> result;
    if (!isLoaded()) { logError() << "[tcxOnnx] run() on unloaded model"; return result; }

    tcxort_clear_feeds();
    for (const auto& kv : namedInputs) {
        const OnnxTensor& t = kv.second;
        std::vector<int32_t> shp(t.shape.begin(), t.shape.end());
        tcxort_set_input(kv.first.c_str(), wasmTypeId(t.type),
                         t.bytes.data(), (int)t.count(), shp.data(), (int)shp.size());
    }
    if (tcxort_run(impl_->session) != 0) return result;

    int n = tcxort_output_count();
    for (int i = 0; i < n; i++) {
        char nameBuf[160] = {0};
        tcxort_output_name(i, nameBuf, (int)sizeof(nameBuf));
        std::string name = nameBuf;
        if (!outputNames.empty() &&
            std::find(outputNames.begin(), outputNames.end(), name) == outputNames.end()) continue;
        OnnxTensor out;
        out.type = fromWasmTypeId(tcxort_output_type(name.c_str()));
        int ndim = tcxort_output_ndim(name.c_str());
        std::vector<int32_t> shp(ndim > 0 ? ndim : 0);
        if (ndim > 0) tcxort_output_shape(name.c_str(), shp.data());
        out.shape.assign(shp.begin(), shp.end());
        int elems = tcxort_output_elems(name.c_str());
        size_t es = out.elementSize(); if (es == 0) es = 4;
        out.bytes.resize((size_t)elems * es);
        if (elems > 0) tcxort_output_data(name.c_str(), out.bytes.data());
        result[name] = std::move(out);
    }
    return result;
}

vector<OnnxModel::TensorInfo> OnnxModel::inputInfo() const { return {}; }
vector<OnnxModel::TensorInfo> OnnxModel::outputInfo() const { return {}; }
void OnnxModel::printModelInfo() const { logNotice() << "[tcxOnnx] (ort-web backend)"; }
void* OnnxModel::nativeSession() { return nullptr; }

#endif

} // namespace tcx
