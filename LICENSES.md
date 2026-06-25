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
Models are not part of this addon. If you ship MediaPipe-derived models
(BlazeFace, FaceMesh, …) note they are Apache-2.0 licensed by Google.
