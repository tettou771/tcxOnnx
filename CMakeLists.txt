# =============================================================================
# tcxOnnx - ONNX Runtime wrapper addon
# =============================================================================
#
# Links a vendored prebuilt ONNX Runtime (in lib/onnxruntime-<platform>-<ver>/).
# The binaries are NOT committed; fetch them with scripts/fetch_onnxruntime.sh
# (CI uses the same script via the addon's setup_script hook).
#
# ORT headers are PRIVATE: tcxOnnx.h exposes no Ort types, so consumers of this
# addon never need the onnxruntime include path. The dylib/so/dll is linked
# PUBLIC so the host app picks up the rpath (matching apps/TrussPhoto).
# =============================================================================

cmake_minimum_required(VERSION 3.16)

set(ADDON_NAME tcxOnnx)

file(GLOB ADDON_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB ADDON_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)

add_library(${ADDON_NAME} STATIC ${ADDON_SOURCES} ${ADDON_HEADERS})
target_include_directories(${ADDON_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${ADDON_NAME} PUBLIC TrussC)

if(EMSCRIPTEN)
    # wasm support is M5: build ORT as a static lib (--build_wasm_static_lib) or
    # fall back to TFLite+XNNPACK for the web target. Not wired yet.
    message(WARNING "tcxOnnx: wasm/Emscripten build is not supported yet (planned, M5)")
else()
    # ----- Locate the vendored prebuilt ONNX Runtime --------------------------
    if(APPLE)
        file(GLOB _ORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-osx-*")
    elseif(WIN32)
        file(GLOB _ORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-win-*")
    elseif(UNIX)
        file(GLOB _ORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-linux-*")
    endif()
    list(SORT _ORT_DIRS)
    list(REVERSE _ORT_DIRS)                  # newest version first if several
    if(_ORT_DIRS)
        list(GET _ORT_DIRS 0 ORT_DIR)
    endif()

    if(NOT ORT_DIR OR NOT EXISTS "${ORT_DIR}/include")
        message(FATAL_ERROR
            "tcxOnnx: no prebuilt ONNX Runtime found under lib/.\n"
            "  Run:  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/fetch_onnxruntime.sh")
    endif()

    target_include_directories(${ADDON_NAME} PRIVATE "${ORT_DIR}/include")
    target_link_directories(${ADDON_NAME} PUBLIC "${ORT_DIR}/lib")
    target_link_libraries(${ADDON_NAME} PUBLIC onnxruntime)

    # Also register the runtime lib for bundling into distributable apps.
    if(COMMAND tc_addon_bundle_file)
        if(APPLE)
            file(GLOB _ORT_DYLIB "${ORT_DIR}/lib/libonnxruntime.*.dylib")
            foreach(_f ${_ORT_DYLIB})
                tc_addon_bundle_file("${_f}")
            endforeach()
        elseif(WIN32)
            file(GLOB _ORT_DLL "${ORT_DIR}/lib/onnxruntime.dll")
            foreach(_f ${_ORT_DLL})
                tc_addon_bundle_file("${_f}")
            endforeach()
        endif()
    endif()

    message(STATUS "tcxOnnx: using ONNX Runtime at ${ORT_DIR}")
endif()
