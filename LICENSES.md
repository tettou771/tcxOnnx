# Licenses

## tcxOnnx

MIT License — Copyright (c) tettou771

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

## Bundled / linked dependencies

### ONNX Runtime
- License: MIT
- Copyright (c) Microsoft Corporation
- https://github.com/microsoft/onnxruntime
- The prebuilt binaries are downloaded at setup time (not redistributed in this
  repository) by `scripts/fetch_onnxruntime.sh`.

### Models
The addon itself is model-agnostic and ships no models for production use. If you
bring MediaPipe-derived models (BlazeFace, FaceMesh, …) note they are Apache-2.0
licensed by Google.

#### Bundled example model: MNIST (`example-basic/bin/data/models/mnist-8.onnx`)
- License: Apache-2.0
- Source: ONNX Model Zoo — https://github.com/onnx/models (repo is Apache-2.0)
- Producer: Microsoft CNTK 2.5.1 (opset 8). ~26 KB.
- Bundled (not fetched) only as a tiny, license-clean fixture so `example-basic`
  runs out of the box. The underlying MNIST dataset is by Y. LeCun et al.
