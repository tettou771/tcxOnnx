// =============================================================================
// tcxOnnx tests - headless behavioral test (no window).
//
// Built and run by the addon CI (TrussC-org/ci-actions build-addon) on every
// push: exit 0 = pass, non-zero = fail. Console only, so it runs on headless
// runners.
//
//   - internal::tensorByteSize: the overflow-safe byte size an input is checked
//     against (negative dims and overflow are rejected).
//   - internal::webElementSize: the per-type element size the web backend
//     copies outputs with (the web bridge itself can't run here).
//   - Model::run() / kick() on the MNIST model of example-basic: an input whose
//     bytes don't match its shape is refused with an error naming it, and no
//     inference runs; a matching input still runs.
// =============================================================================

#include <tcxOnnx.h>
#include <tcxOnnxInternal.h>

#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

using namespace std;
using namespace tc;
using namespace tcx;

static int g_pass = 0, g_fail = 0;
static void check(const char* name, bool ok) {
    printf("%-64s %s\n", name, ok ? "PASS" : "FAIL");
    fflush(stdout);   // flush each line so CI logs survive a later crash
    ok ? ++g_pass : ++g_fail;
}

// Errors logged since the last clear, to check what a refused input reports.
static vector<string> g_errors;
static bool errorMentions(const vector<string>& parts) {
    for (const auto& e : g_errors) {
        bool all = true;
        for (const auto& p : parts) all = all && e.find(p) != string::npos;
        if (all) return true;
    }
    return false;
}

static void testTensorByteSize() {
    using onnx::internal::tensorByteSize;
    size_t b = 12345;
    check("byteSize: {1,1,28,28} x4 = 3136", tensorByteSize({1, 1, 28, 28}, 4, b) && b == 3136);
    b = 12345;
    check("byteSize: empty shape = 0 (as Tensor::count())", tensorByteSize({}, 4, b) && b == 0);
    b = 12345;
    check("byteSize: a zero dim = 0", tensorByteSize({2, 0, 3}, 8, b) && b == 0);
    b = 12345;
    check("byteSize: a negative dim is rejected", !tensorByteSize({1, -1, 3}, 4, b) && b == 12345);
    b = 12345;
    check("byteSize: a lone -1 dim of 1-byte elements is rejected", !tensorByteSize({-1}, 1, b) && b == 12345);
    const int64_t big = numeric_limits<int64_t>::max();
    b = 12345;
    check("byteSize: dim product overflow is rejected", !tensorByteSize({big, 3}, 1, b) && b == 12345);
    b = 12345;
    check("byteSize: 2^32 x 2^32 overflow is rejected",
          !tensorByteSize({int64_t(1) << 32, int64_t(1) << 32}, 1, b) && b == 12345);
    b = 12345;
    check("byteSize: element size overflow is rejected", !tensorByteSize({big}, 4, b) && b == 12345);
}

static void testWebElementSize() {
    using onnx::internal::webElementSize;
    const struct { const char* type; size_t size; } table[] = {
        {"bool", 1},    {"int8", 1},    {"uint8", 1},
        {"float16", 2}, {"int16", 2},   {"uint16", 2},
        {"float32", 4}, {"int32", 4},   {"uint32", 4},
        {"float64", 8}, {"int64", 8},   {"uint64", 8},
        {"string", 0},  {"int4", 0},    {"uint4", 0},
        {"", 0},        {"Float32", 0}, {"double", 0},
    };
    bool ok = true;
    for (const auto& e : table) {
        const size_t got = webElementSize(e.type);
        if (got != e.size) {
            printf("  webElementSize(\"%s\") = %zu, expected %zu\n", e.type, got, e.size);
            ok = false;
        }
    }
    check("webElementSize: ort-web type table", ok);
}

static void testModel() {
    // The MNIST model shipped with example-basic (input [1,1,28,28] float32).
    // __FILE__ can be relative (CI builds from tests/), so also try the
    // paths relative to the working directory and to the executable.
    const filesystem::path rel = filesystem::path("example-basic") / "bin" / "data" / "models" / "mnist-8.onnx";
    const filesystem::path candidates[] = {
        filesystem::path(__FILE__).parent_path() / ".." / ".." / rel,
        filesystem::path("..") / rel,
        getExecutableDir() / ".." / ".." / rel,
    };
    filesystem::path modelPath = candidates[0];
    for (const auto& c : candidates) {
        error_code ec;
        if (filesystem::exists(c, ec)) { modelPath = c; break; }
    }
    onnx::Model model;
    const bool loaded = model.load(pathToUtf8(modelPath));
    check("model: MNIST loads", loaded);
    if (!loaded) return;
    const auto inNames = model.inputNames();
    const auto outNames = model.outputNames();
    check("model: one input, one output", inNames.size() == 1 && outNames.size() == 1);
    if (inNames.size() != 1 || outNames.size() != 1) return;
    const string in = inNames[0], out = outNames[0];
    const vector<int64_t> shape = {1, 1, 28, 28};

    // A matching input runs.
    onnx::Result r = model.run({{in, onnx::Tensor::f32(vector<float>(784, 0.5f), shape)}});
    check("run: matching input returns the output", r.get(out).asFloat().size() == 10);

    // Too few bytes: ORT would read 3136 bytes from a 48-byte buffer.
    g_errors.clear();
    r = model.run({{in, onnx::Tensor::f32(vector<float>(12, 0.5f), shape)}});
    check("run: short input returns an empty Result", r.empty());
    check("run: the error names the input and both sizes", errorMentions({"'" + in + "'", "48", "3136"}));

    g_errors.clear();
    r = model.run({{in, onnx::Tensor::f32(vector<float>(785, 0.5f), shape)}});
    check("run: long input returns an empty Result", r.empty());
    check("run: the long input is reported", errorMentions({"'" + in + "'", "3140", "3136"}));

    g_errors.clear();
    r = model.run({{in, onnx::Tensor::f32(vector<float>(784, 0.5f), {1, 1, -28, 28})}});
    check("run: negative dim returns an empty Result", r.empty());
    check("run: the negative dim is reported", errorMentions({"'" + in + "'", "invalid shape"}));

    // Every input is checked before any runs; the error names the bad one.
    g_errors.clear();
    r = model.run({{in, onnx::Tensor::f32(vector<float>(784, 0.5f), shape)},
                   {"zz_extra", onnx::Tensor::i64({1, 2}, {3})}});
    check("run: one bad input of two returns an empty Result", r.empty());
    check("run: the error names the bad input", errorMentions({"'zz_extra'", "16", "24"}));

    // kick(): a bad input starts nothing, so no result appears.
    model.takeResult();
    g_errors.clear();
    model.kick({{in, onnx::Tensor::f32(vector<float>(12, 0.5f), shape)}});
    check("kick: short input starts no inference", !model.hasResult());
    check("kick: the error names the input", errorMentions({"'" + in + "'", "48", "3136"}));
    model.kick(onnx::Tensor::f32(vector<float>(12, 0.5f), shape));
    check("kick(single): short input starts no inference", !model.hasResult());

    model.kick({{in, onnx::Tensor::f32(vector<float>(784, 0.5f), shape)}});
    check("kick: matching input has a result", model.hasResult());
    check("kick: the result has the output", model.takeResult().get(out).asFloat().size() == 10);
}

int main() {
    auto errorListener = getLogger().onLog.listen([](LogEventArgs& e) {
        if (e.level == LogLevel::Error) g_errors.push_back(e.message);
    });

    testTensorByteSize();
    testWebElementSize();
    testModel();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
