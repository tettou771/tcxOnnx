#include "tcApp.h"
#include <filesystem>

namespace fs = std::filesystem;

// Find the first .onnx file under the app's data dir (resolved via getDataPath,
// like oF's ofToDataPath — independent of the launch cwd).
static std::string findFirstModel() {
    for (const std::string& rel : {std::string("models"), std::string("")}) {
        std::string dir = getDataPath(rel);
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) continue;
        for (auto& e : fs::directory_iterator(dir, ec)) {
            if (e.path().extension() == ".onnx") return e.path().string();
        }
    }
    return "";
}

void tcApp::setup() {
    std::string path = findFirstModel();
    if (path.empty()) {
        status_ = "Drop a .onnx into data/models/ and relaunch.";
        logWarning() << "[tcxOnnx example] " << status_;
        return;
    }

    OnnxModel::Options opts;
    opts.verbose = true;
    loaded_ = model_.load(path, opts);
    if (!loaded_) {
        status_ = "failed to load: " + path;
        return;
    }

    model_.printModelInfo();

    // Build all-zero dummy inputs matching each input's shape (dynamic dims -> 1)
    // and run once, just to prove the full path works end to end.
    std::map<std::string, OnnxTensor> inputs;
    for (const auto& in : model_.inputInfo()) {
        std::vector<int64_t> shape = in.shape;
        for (auto& d : shape) if (d < 0) d = 1;
        OnnxTensor t;
        t.shape = shape;
        t.type = (in.type == OnnxTensor::Type::Other) ? OnnxTensor::Type::Float32 : in.type;
        size_t bytes = t.count() * t.elementSize();
        t.bytes.assign(bytes, 0);
        inputs[in.name] = std::move(t);
    }

    auto outputs = model_.run(inputs);
    status_ = "loaded " + fs::path(path).filename().string() +
              " | ran ok, " + std::to_string(outputs.size()) + " outputs";
    for (auto& kv : outputs) {
        std::string shp;
        for (size_t i = 0; i < kv.second.shape.size(); i++) {
            if (i) shp += ",";
            shp += std::to_string(kv.second.shape[i]);
        }
        logNotice() << "[tcxOnnx example] output " << kv.first << " [" << shp << "]";
    }
}

void tcApp::update() {}

void tcApp::draw() {
    clear(0.1f, 0.1f, 0.12f);
    setColor(1.0f, 1.0f, 1.0f);
    drawBitmapString(status_, 20, 30);
}
