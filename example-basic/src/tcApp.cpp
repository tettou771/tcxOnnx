#include "tcApp.h"
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

// ---- Layout (window is 800x600) --------------------------------------------
static constexpr float kCx = 40,  kCy = 120, kCs = 420;   // canvas: x, y, size
static constexpr float kBx = 520, kBy = 120;              // probability bars origin

// Find the first .onnx file under the app's data dir (resolved via getDataPath,
// independent of the launch cwd).
static std::string findFirstModel() {
    for (const std::string& rel : {std::string("models"), std::string("")}) {
        std::string dir = getDataPath(rel);
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) continue;
        for (auto& e : fs::directory_iterator(dir, ec))
            if (e.path().extension() == ".onnx") return e.path().string();
    }
    return "";
}

void tcApp::setup() {
    std::string path = findFirstModel();
    if (path.empty()) {
        status_ = "Drop an MNIST .onnx into data/models/ and relaunch.";
        logWarning() << "[example] " << status_;
        return;
    }

    onnx::Model::Options opts;
    opts.verbose = true;
    loaded_ = model_.load(path, opts);
    if (!loaded_) { status_ = "failed to load: " + path; return; }
    model_.printModelInfo();

    // MNIST has a single input and a single output; grab their names so the rest
    // of the code stays model-name-agnostic.
    auto ins = model_.inputNames(), outs = model_.outputNames();
    if (!ins.empty())  inputName_  = ins.front();
    if (!outs.empty()) outputName_ = outs.front();
    status_ = "loaded " + fs::path(path).filename().string() + " - draw a digit";
}

void tcApp::update() {
    if (loaded_ && dirty_) { infer(); dirty_ = false; }
}

// Soft circular brush so strokes resemble MNIST's anti-aliased pen.
void tcApp::paintAt(float mx, float my) {
    float cell = kCs / N;
    float gx = (mx - kCx) / cell, gy = (my - kCy) / cell;   // grid-space coords
    if (gx < -2 || gy < -2 || gx > N + 2 || gy > N + 2) return;
    const float radius = 1.3f;
    int y0 = std::max(0, (int)std::floor(gy - radius)), y1 = std::min(N - 1, (int)std::ceil(gy + radius));
    int x0 = std::max(0, (int)std::floor(gx - radius)), x1 = std::min(N - 1, (int)std::ceil(gx + radius));
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) {
            float d = std::hypot((x + 0.5f) - gx, (y + 0.5f) - gy);
            float v = std::max(0.0f, 1.0f - d / radius);
            float& c = canvas_[y * N + x];
            c = std::min(1.0f, c + v);
        }
    dirty_ = true;
}

void tcApp::clearCanvas() {
    std::fill(canvas_.begin(), canvas_.end(), 0.0f);
    predicted_ = -1; confidence_ = 0.0f;
    std::fill(probs_.begin(), probs_.end(), 0.0f);
}

void tcApp::infer() {
    if (inputName_.empty()) return;

    // Preprocess: the 28x28 canvas (0..1) -> NCHW float tensor in [0,255], which
    // is what the ONNX model-zoo MNIST model expects.
    std::vector<float> in(N * N);
    for (int i = 0; i < N * N; i++) in[i] = canvas_[i] * 255.0f;

    // Single-input convenience: run(Tensor) uses the model's sole input name.
    onnx::Result r = model_.run(onnx::Tensor::f32(std::move(in), {1, 1, N, N}));

    std::vector<float> logits = r.get(outputName_).asFloat();   // safe even if missing
    if (logits.size() < 10) { predicted_ = -1; return; }

    // Softmax for display.
    float mx = *std::max_element(logits.begin(), logits.begin() + 10), sum = 0;
    for (int i = 0; i < 10; i++) { probs_[i] = std::exp(logits[i] - mx); sum += probs_[i]; }
    for (int i = 0; i < 10; i++) probs_[i] /= (sum > 0 ? sum : 1);
    predicted_ = (int)(std::max_element(probs_.begin(), probs_.end()) - probs_.begin());
    confidence_ = probs_[predicted_];
}

void tcApp::draw() {
    clear(0.1f, 0.1f, 0.12f);

    // ---- header ----
    setColor(1, 1, 1);
    drawBitmapString(status_, 20, 30);
    drawBitmapString("left-drag: draw   |   right-click / 'c': clear", 20, 52);

    // ---- canvas (28x28 cells) ----
    setColor(0, 0, 0);
    drawRect(kCx, kCy, kCs, kCs);
    float cell = kCs / N;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            float v = canvas_[y * N + x];
            if (v <= 0.001f) continue;
            setColor(v, v, v);
            drawRect(kCx + x * cell, kCy + y * cell, cell, cell);
        }
    setColor(0.4f, 0.4f, 0.45f);   // frame
    drawRect(kCx, kCy, kCs, 1); drawRect(kCx, kCy + kCs, kCs, 1);
    drawRect(kCx, kCy, 1, kCs); drawRect(kCx + kCs, kCy, 1, kCs);

    // ---- prediction + probability bars ----
    if (predicted_ >= 0) {
        setColor(0.4f, 1.0f, 0.6f);
        drawBitmapString("prediction: " + std::to_string(predicted_) +
                         "   (" + std::to_string((int)std::round(confidence_ * 100)) + "%)",
                         kBx, kCy - 20);
    }
    for (int d = 0; d < 10; d++) {
        float by = kBy + d * 38.0f;
        setColor(0.7f, 0.7f, 0.75f);
        drawBitmapString(std::to_string(d), kBx - 18, by + 16);
        setColor(0.2f, 0.2f, 0.25f);
        drawRect(kBx, by, 230, 22);
        bool best = (d == predicted_);
        if (best) setColor(0.4f, 1.0f, 0.6f); else setColor(0.45f, 0.6f, 0.95f);
        drawRect(kBx, by, 230 * probs_[d], 22);
    }
}

void tcApp::keyPressed(int key) {
    if (key == 'c' || key == 'C') clearCanvas();
}

void tcApp::mousePressed(const MouseEventArgs& e) {
    if (e.button == MOUSE_BUTTON_RIGHT) { clearCanvas(); return; }
    paintAt(e.x, e.y);
}

void tcApp::mouseDragged(const MouseDragEventArgs& e) {
    if (e.button == MOUSE_BUTTON_RIGHT) return;
    paintAt(e.x, e.y);
}
