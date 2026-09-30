#pragma once

// =============================================================================
// tcxOnnxInternal.h - pure helpers used by tcxOnnx.cpp, declared here only so
// tests/ can check them. Not part of the addon's API; may change at any time.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tcx::onnx::internal {

// Bytes that a tensor of `shape` takes with `elementSize`-byte elements,
// counting elements the way Tensor::count() does (an empty shape counts as 0).
// Returns false, leaving `outBytes` untouched, when a dim is negative or the
// size doesn't fit in size_t.
bool tensorByteSize(const std::vector<int64_t>& shape, size_t elementSize, size_t& outBytes);

} // namespace tcx::onnx::internal
