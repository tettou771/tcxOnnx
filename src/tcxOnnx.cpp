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
// WASM backend (onnxruntime-web via a NON-BLOCKING EM_JS bridge).
//
// CRITICAL: ort-web's create()/run() are async. We must NOT await them from the
// frame callback — Asyncify would unwind sokol_app's main loop and it never
// resumes (the app freezes after one frame). So this bridge is fire-and-forget:
// load()/run() kick the JS promise and return immediately; results are picked up
// on a later frame by polling. Detection lags a few frames, which is fine.
// State lives on Module.__ortx; models are read from the preloaded FS (/data).
//   sessions[h] = { sess, feeds, inflight, hasResult, result }
// =============================================================================

// One-time, non-blocking: inject ort-web from a CDN. Sets ready on script load.
EM_JS(void, tcxort_init, (), {
    if (!Module.__ortx) Module.__ortx = { ready:false, loading:false, sessions:[] };
    var X = Module.__ortx;
    if (X.ready || X.loading) return;
    var cfg = function(){
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
        ort.env.wasm.numThreads = 1;   // gh-pages can't set COOP/COEP -> single thread
        X.ready = true;
    };
    if (typeof ort !== 'undefined') { cfg(); return; }
    X.loading = true;
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.js';
    s.onload = cfg;
    s.onerror = function(){ console.error('ort-web load failed'); X.loading = false; };
    document.head.appendChild(s);
});

EM_JS(int, tcxort_ready, (), { return (Module.__ortx && Module.__ortx.ready) ? 1 : 0; });

// Reserve a session slot and copy the model bytes. The actual async create is
// DEFERRED until ort-web has loaded (see tcxort_session_ready) — calling
// ort.* before the CDN script lands throws "ort is not defined" and an uncaught
// error out of the frame callback would freeze the loop.
EM_JS(int, tcxort_create_kick, (const unsigned char* data, int len), {
    var slot = { sess:null, bytes:HEAPU8.slice(data, data + len), creating:false,
                 feeds:{}, inflight:false, hasResult:false, result:null };
    Module.__ortx.sessions.push(slot);
    return Module.__ortx.sessions.length - 1;
});

// Polled each frame by isLoaded(): once ort-web is ready, kick the deferred
// create; returns 1 only when the session object actually exists.
EM_JS(int, tcxort_session_ready, (int h), {
    var slot = Module.__ortx.sessions[h];
    if (!slot) return 0;
    if (slot.sess) return 1;
    if (Module.__ortx.ready && slot.bytes && !slot.creating) {
        slot.creating = true;
        ort.InferenceSession.create(slot.bytes, { executionProviders:['wasm'] })
            .then(function(s){ slot.sess = s; slot.bytes = null; })
            .catch(function(e){ console.error('ort create failed', e); slot.creating = false; });
    }
    return 0;
});

EM_JS(void, tcxort_clear_feeds, (int h), { var slot = Module.__ortx.sessions[h]; if (slot) slot.feeds = {}; });

// Stage one input. type: 0=f32, 1=i64, 2=i32, 3=u8.
EM_JS(void, tcxort_set_input, (int h, const char* namePtr, int type, const void* dataPtr, int count, const int* shapePtr, int ndim), {
    var slot = Module.__ortx.sessions[h]; if (!slot) return;
    var name = UTF8ToString(namePtr);
    var dims = [];
    for (var i = 0; i < ndim; i++) dims.push(HEAP32[(shapePtr >> 2) + i]);
    var t;
    if (type === 0)      t = new ort.Tensor('float32', HEAPF32.slice(dataPtr >> 2, (dataPtr >> 2) + count), dims);
    else if (type === 1) t = new ort.Tensor('int64', new BigInt64Array(HEAP8.buffer, dataPtr, count).slice(), dims);
    else if (type === 2) t = new ort.Tensor('int32', HEAP32.slice(dataPtr >> 2, (dataPtr >> 2) + count), dims);
    else                 t = new ort.Tensor('uint8', HEAPU8.slice(dataPtr, dataPtr + count), dims);
    slot.feeds[name] = t;
});

// Kick an async run if the session is idle (non-blocking). Result stored on resolve.
EM_JS(void, tcxort_kick, (int h), {
    var slot = Module.__ortx.sessions[h];
    if (!slot || !slot.sess || slot.inflight) return;
    slot.inflight = true;
    var feeds = slot.feeds;   // clear_feeds rebinds slot.feeds; this run keeps the old object
    slot.sess.run(feeds)
        .then(function(r){ slot.result = r; slot.hasResult = true; slot.inflight = false; })
        .catch(function(e){ console.error('ort run failed', e); slot.inflight = false; });
});

EM_JS(int, tcxort_has_result, (int h), {
    var slot = Module.__ortx.sessions[h];
    return (slot && slot.hasResult) ? 1 : 0;
});

EM_JS(int, tcxort_output_count, (int h), {
    var slot = Module.__ortx.sessions[h];
    return (slot && slot.result) ? Object.keys(slot.result).length : 0;
});

EM_JS(int, tcxort_output_name, (int h, int idx, char* buf, int buflen), {
    var slot = Module.__ortx.sessions[h];
    var k = (slot && slot.result) ? Object.keys(slot.result)[idx] : undefined;
    if (k === undefined) { if (buflen) HEAPU8[buf] = 0; return 0; }
    stringToUTF8(k, buf, buflen);
    return lengthBytesUTF8(k);
});

EM_JS(int, tcxort_output_type, (int h, const char* namePtr), {
    var slot = Module.__ortx.sessions[h];
    var o = (slot && slot.result) ? slot.result[UTF8ToString(namePtr)] : null;
    if (!o) return 4;
    if (o.type === 'float32') return 0;
    if (o.type === 'int64')   return 1;
    if (o.type === 'int32')   return 2;
    if (o.type === 'uint8')   return 3;
    return 4;
});

EM_JS(int, tcxort_output_ndim, (int h, const char* namePtr), {
    var slot = Module.__ortx.sessions[h];
    var o = (slot && slot.result) ? slot.result[UTF8ToString(namePtr)] : null;
    return o ? o.dims.length : 0;
});

EM_JS(void, tcxort_output_shape, (int h, const char* namePtr, int* outShape), {
    var slot = Module.__ortx.sessions[h];
    var o = (slot && slot.result) ? slot.result[UTF8ToString(namePtr)] : null;
    if (!o) return;
    for (var i = 0; i < o.dims.length; i++) HEAP32[(outShape >> 2) + i] = o.dims[i];
});

EM_JS(int, tcxort_output_elems, (int h, const char* namePtr), {
    var slot = Module.__ortx.sessions[h];
    var o = (slot && slot.result) ? slot.result[UTF8ToString(namePtr)] : null;
    return o ? o.data.length : 0;
});

EM_JS(void, tcxort_output_data, (int h, const char* namePtr, void* dst), {
    var slot = Module.__ortx.sessions[h];
    var o = (slot && slot.result) ? slot.result[UTF8ToString(namePtr)] : null;
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
    tcxort_init();   // non-blocking; ort-web loads from CDN in the background
    std::ifstream f(modelPath, std::ios::binary);
    if (!f) { logError() << "[tcxOnnx] cannot open " << modelPath; return false; }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    impl_->session = tcxort_create_kick(bytes.data(), (int)bytes.size());
    if (opts.verbose) logNotice() << "[tcxOnnx] loading (ort-web, async): " << modelPath;
    return impl_->session >= 0;
}

// The session loads asynchronously on web; isLoaded() reflects real readiness so
// callers naturally skip inference until ort-web + the session are ready.
bool OnnxModel::isLoaded() const {
    return impl_ && impl_->session >= 0 && tcxort_ready() && tcxort_session_ready(impl_->session);
}
void OnnxModel::unload() { if (impl_) impl_->session = -1; }

static int wasmTypeId(OnnxTensor::Type t) {
    switch (t) { case OnnxTensor::Type::Float32: return 0; case OnnxTensor::Type::Int64: return 1;
                 case OnnxTensor::Type::Int32: return 2; case OnnxTensor::Type::UInt8: return 3; default: return 0; }
}
static OnnxTensor::Type fromWasmTypeId(int t) {
    switch (t) { case 0: return OnnxTensor::Type::Float32; case 1: return OnnxTensor::Type::Int64;
                 case 2: return OnnxTensor::Type::Int32; case 3: return OnnxTensor::Type::UInt8; default: return OnnxTensor::Type::Other; }
}

// Non-blocking: stage inputs + kick a run, then return the MOST RECENT completed
// result (empty until the first one lands). The outputs lag the inputs by one
// inference; for per-frame detection that few-frame latency is acceptable.
map<string, OnnxTensor> OnnxModel::run(const map<string, OnnxTensor>& namedInputs, const vector<string>& outputNames) {
    map<string, OnnxTensor> result;
    if (!isLoaded()) return result;   // ort-web / session still loading
    const int h = impl_->session;

    tcxort_clear_feeds(h);
    for (const auto& kv : namedInputs) {
        const OnnxTensor& t = kv.second;
        std::vector<int32_t> shp(t.shape.begin(), t.shape.end());
        tcxort_set_input(h, kv.first.c_str(), wasmTypeId(t.type),
                         t.bytes.data(), (int)t.count(), shp.data(), (int)shp.size());
    }
    tcxort_kick(h);                        // fire-and-forget
    if (!tcxort_has_result(h)) return result;   // not ready yet -> caller keeps last

    int n = tcxort_output_count(h);
    for (int i = 0; i < n; i++) {
        char nameBuf[160] = {0};
        tcxort_output_name(h, i, nameBuf, (int)sizeof(nameBuf));
        std::string name = nameBuf;
        if (!outputNames.empty() &&
            std::find(outputNames.begin(), outputNames.end(), name) == outputNames.end()) continue;
        OnnxTensor out;
        out.type = fromWasmTypeId(tcxort_output_type(h, name.c_str()));
        int ndim = tcxort_output_ndim(h, name.c_str());
        std::vector<int32_t> shp(ndim > 0 ? ndim : 0);
        if (ndim > 0) tcxort_output_shape(h, name.c_str(), shp.data());
        out.shape.assign(shp.begin(), shp.end());
        int elems = tcxort_output_elems(h, name.c_str());
        size_t es = out.elementSize(); if (es == 0) es = 4;
        out.bytes.resize((size_t)elems * es);
        if (elems > 0) tcxort_output_data(h, name.c_str(), out.bytes.data());
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
