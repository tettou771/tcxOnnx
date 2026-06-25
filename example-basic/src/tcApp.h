#pragma once

#include <TrussC.h>
#include <tcxOnnx.h>

using namespace tc;
using namespace tcx;

class tcApp : public App {
public:
    void setup() override;
    void update() override;
    void draw() override;

private:
    OnnxModel model_;
    bool loaded_ = false;
    std::string status_ = "no model loaded";
};
