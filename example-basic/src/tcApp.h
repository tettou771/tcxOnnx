#pragma once

#include <TrussC.h>
#include <tcxOnnx.h>

using namespace tc;
using namespace tcx;

// =============================================================================
// tcxOnnx example-basic - MNIST handwritten-digit recognition.
//
// Draw a digit with the mouse; the app preprocesses the 28x28 canvas and runs
// the bundled MNIST model every frame the canvas changes. It is a small, honest
// end-to-end usage reference: load() -> run(Tensor) single-input convenience ->
// Result::get() -> Tensor::asFloat().  Right-click or 'c' clears.
//
// Note: this small MNIST CNN is overconfident — softmax always sums to 1 over 10
// classes with no "not a digit" option, so scribbles still get a ~100% pick. That
// is the model's nature, not a bug in the pipeline.
// =============================================================================
class tcApp : public App {
public:
    void setup() override;
    void update() override;
    void draw() override;

    void keyPressed(int key) override;
    void mousePressed(const MouseEventArgs& e) override;
    void mouseDragged(const MouseDragEventArgs& e) override;

private:
    static constexpr int N = 28;                  // MNIST is 28x28

    onnx::Model model_;
    bool loaded_ = false;
    std::string status_ = "no model loaded";
    std::string inputName_, outputName_;          // resolved from the model

    std::vector<float> canvas_ = std::vector<float>(N * N, 0.0f);  // 0..1 grayscale
    bool dirty_ = false;                          // canvas changed -> re-infer
    int predicted_ = -1;
    float confidence_ = 0.0f;
    std::vector<float> probs_ = std::vector<float>(10, 0.0f);

    void clearCanvas();
    void paintAt(float mx, float my);             // soft brush into canvas_
    void infer();                                 // canvas_ -> model -> probs_
};
