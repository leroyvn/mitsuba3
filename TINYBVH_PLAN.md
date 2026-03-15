# Plan: Replace Mitsuba's k-d Tree and Embree with TinyBVH

## Context

Mitsuba 3 ships with two CPU-side acceleration structures:
- **Built-in k-d tree** (`ShapeKDTree`, currently active in Eradiate — `MI_ENABLE_EMBREE=OFF`)
- **Embree** (Intel, optional, currently disabled)

The goal is to **replace both** with **TinyBVH** (https://github.com/jbikker/tinybvh), a header-only, MIT-licensed BVH library. TinyBVH supports both single and double precision, making it a unified replacement for all Mitsuba variants including `*_double` ones.

This work is entirely within the `ext/mitsuba` submodule. Eradiate's Python code is not touched. TinyBVH is added as a git submodule.

---

## Architecture of Mitsuba's Acceleration Structure System

The backend is selected at build time via `scene.cpp`:

```cpp
#if defined(MI_ENABLE_EMBREE)
#  include "scene_embree.inl"
#else
#  include <mitsuba/render/kdtree.h>
#  include "scene_native.inl"
#endif
```

Each backend must implement three functions in its `.inl` file:

| Function | Purpose |
|---|---|
| `accel_init_cpu(props, shapes)` | Build the acceleration structure |
| `accel_parameters_changed_cpu(shapes)` | Rebuild/refit when geometry changes |
| `ray_intersect_preliminary_cpu(ray, prim_types, active)` | Trace rays (returns `PreliminaryIntersection`) |

The backend stores its state in a per-`Scene` struct held as `void* m_accel`.

---

## Implementation Status

### ✅ Completed

- **Step 1**: TinyBVH added as git submodule at `ext/tinybvh`
- **Step 2**: CMake option `MI_ENABLE_TINYBVH` added to `CMakeLists.txt` and
  `src/render/CMakeLists.txt`; pixi task `kernel-configure-tinybvh` added to
  `pyproject.toml`
- **Step 3**: `scene.cpp` `#elif MI_ENABLE_TINYBVH` branch added
- **Step 4**: `scene_tinybvh.inl` implemented:
  - `TinyBVHState` struct with precision-dispatched `BVHType` and persistent
    `vertices` buffer (TinyBVH stores a raw pointer — buffer must outlive the BVH)
  - `accel_parameters_changed_cpu`: builds BVH from mesh triangles + 12-triangle
    bounding-box proxies for analytic shapes; zero-extent axes inflated by 1e-4
    to avoid degenerate proxy triangles
  - `ray_intersect_preliminary_cpu`: scalar path calls `bvh.Intersect()`
    directly; LLVM path uses `tinybvh_trace_func_wrapper` registered via
    `jit_llvm_ray_trace` (mirrors `kdtree_trace_func_wrapper`)
  - Empty-scene guard: `bvhNode` is null until `Build()` is called; guarded in
    both scalar path and LLVM wrapper
  - `ray_test_cpu`: scalar delegates to `ray_intersect_preliminary_cpu`; LLVM
    uses shadow-ray flag
- **ShapeGroup BLAS**: `ShapeGroup` now builds its own per-group TinyBVH BLAS
  (previously always used `ShapeKDTree`). Double-precision variants use
  `BVH_Double` throughout. Vertex data stored in `m_bvh_vertices` member.
- **mradiancemeter fix**: `sensor_index` clamped to `[0, m_sensor_count - 1]`
  to prevent out-of-bounds gather when `sample_ray_differential` adds +1/width
  offset for the last pixel.

### ❌ Not yet done

- **TLAS / Instance shapes**: `Instance` shapes (`shapegroup` + `instance`)
  currently fall through to the default path. TinyBVH supports
  `BuildTLAS(BLASInstance*)`. Each `ShapeGroup` BLAS is built; the `Instance`
  plugin needs to register `BLASInstance` entries with the per-instance
  transform (mirrors Embree's `rtcSetGeometryTransform()` approach).

---

## Implementation Plan (original)

### Step 1 — Add TinyBVH as a git submodule

Inside the Mitsuba submodule:

```bash
cd ext/mitsuba/ext
git submodule add https://github.com/jbikker/tinybvh tinybvh
```

Only `tiny_bvh.h` is needed at build time (header-only library).

---

### Step 2 — CMake configuration

#### `ext/mitsuba/CMakeLists.txt`

Add a new option alongside `MI_ENABLE_EMBREE`:

```cmake
option(MI_ENABLE_TINYBVH "Use TinyBVH for ray tracing operations?" OFF)

if (MI_ENABLE_TINYBVH)
    if (MI_ENABLE_EMBREE)
        message(FATAL_ERROR "MI_ENABLE_TINYBVH and MI_ENABLE_EMBREE are mutually exclusive.")
    endif()
    add_definitions(-DMI_ENABLE_TINYBVH=1)
    include_directories(${CMAKE_SOURCE_DIR}/ext/tinybvh)
    message(STATUS "Mitsuba: using TinyBVH for CPU ray tracing.")
endif()
```

#### `ext/mitsuba/src/render/CMakeLists.txt`

Exclude `kdtree.cpp` when TinyBVH is active (mirrors the existing Embree exclusion):

```cmake
if (NOT MI_ENABLE_EMBREE AND NOT MI_ENABLE_TINYBVH)
    target_sources(mitsuba-render PRIVATE kdtree.cpp)
endif()
```

---

### Step 3 — Update `scene.cpp` conditional

**File:** `ext/mitsuba/src/render/scene.cpp`

```cpp
#if defined(MI_ENABLE_EMBREE)
#  include "scene_embree.inl"
#elif defined(MI_ENABLE_TINYBVH)
#  include "scene_tinybvh.inl"
#else
#  include <mitsuba/render/kdtree.h>
#  include "scene_native.inl"
#endif
```

---

### Step 4 — Create `scene_tinybvh.inl`

**File:** `ext/mitsuba/src/render/scene_tinybvh.inl`

This file mirrors the structure of `scene_embree.inl`.

#### 4a. State struct

```cpp
#define TINYBVH_IMPLEMENTATION
#include <tiny_bvh.h>

template <typename Float>
struct TinyBVHState {
    // Select precision-appropriate BVH type based on scalar float type
    using ScalarF = dr::scalar_t<Float>;
    using BVHType = std::conditional_t<std::is_same_v<ScalarF, double>,
                                       tinybvh::BVH_Double,  // double-precision BVH
                                       tinybvh::BVH>;        // single-precision BVH

    BVHType bvh;

    // Vertex data backing the BVH — TinyBVH stores a pointer, not a copy.
    // Must outlive the BVH object.
    using VertexType = std::conditional_t<std::is_same_v<ScalarF, double>,
                                          tinybvh::bvhdbl3, tinybvh::bvhvec4>;
    std::vector<VertexType> vertices;

    DynamicBuffer<UInt32> shapes_registry_ids;         // JIT-managed shape pointer table
    std::vector<uint32_t> prim_to_shape;               // Maps BVH prim index → shape index in m_shapes
    std::vector<uint32_t> prim_to_local;               // Maps BVH prim index → prim index within shape
    std::vector<bool>     prim_is_mesh;                // true = mesh triangle; false = bbox proxy
};
```

#### 4b. Shape geometry collection (`accel_init_cpu`)

**Analytic shapes** (sphere, disk, cylinder) do **not** provide triangle geometry. The Embree backend handles this via a `RTC_GEOMETRY_TYPE_USER` registration in the base `Shape::embree_geometry()`, which installs custom `embree_intersect` / `embree_occluded` callbacks. For TinyBVH we follow the same pattern conceptually:

- Register each analytic shape's **bounding box** as a proxy entry (12 triangles, 2 per face, for a watertight proxy mesh).
- On BVH hit against a proxy primitive, **call the shape's exact `ray_intersect_preliminary()`** — the same fallback logic that Embree's USER geometry callbacks invoke.
- Zero-extent bbox axes (flat shapes like `arectangle`, `disk`) are inflated by `PROXY_INFLATE = 1e-4` to prevent degenerate proxy triangles.

**Triangle meshes** contribute their actual vertex/face data directly, using `mesh->vertex_positions_buffer()` and `mesh->faces_buffer()`.

#### 4c. `accel_parameters_changed_cpu`

Rebuilds from scratch on each call (clear state, repeat 4b). `bvh.Refit()` could be used for near-rigid deformations in the future.

#### 4d. `ray_intersect_preliminary_cpu` — JIT callback (scalar and LLVM paths)

Both paths use `tinybvh_trace_func_wrapper<Float, Spectrum, ShadowRay, Width>`, mirroring `kdtree_trace_func_wrapper` in `scene_native.inl`. Width is selected at init time via `jit_llvm_vector_width()` (1/4/8/16).

**Double precision:** `BVHType = tinybvh::BVH_Double` when `ScalarF == double`; full double-precision traversal with no downcast, unlike Embree.

#### 4e. Instances (not yet implemented)

TinyBVH supports TLAS via `BuildTLAS(BLASInstance*)`. Each `ShapeGroup` builds its own BLAS; `Instance` shapes register `BLASInstance` entries with the per-instance transform. This mirrors Embree's IAS approach where `instance.cpp` calls `rtcSetGeometryTransform()`.

---

### Step 5 — `pixi.toml`: add build task ✅

```toml
[tasks]
kernel-configure-tinybvh = { cmd = "cmake -S ext/mitsuba -B ext/mitsuba/build -DCMAKE_BUILD_TYPE=Release -GNinja --preset eradiate -DMI_ENABLE_TINYBVH=ON" }
```

---

## Critical Files

| File | Change | Status |
|---|---|---|
| `ext/tinybvh/` | **New** — git submodule | ✅ |
| `CMakeLists.txt` | Add `MI_ENABLE_TINYBVH` option | ✅ |
| `src/render/CMakeLists.txt` | Exclude `kdtree.cpp` when TinyBVH active | ✅ |
| `src/render/scene.cpp` | Add `#elif MI_ENABLE_TINYBVH` branch | ✅ |
| `src/render/scene_tinybvh.inl` | **New** — backend implementation | ✅ |
| `include/mitsuba/render/shapegroup.h` | TinyBVH BLAS members | ✅ |
| `src/render/shapegroup.cpp` | Build per-group TinyBVH BLAS | ✅ |
| `src/eradiate_plugins/sensors/mradiancemeter.cpp` | Clamp sensor_index | ✅ |
| `instance.cpp` (future) | Register BLASInstance for TLAS | ❌ |

---

## Key Challenges & Mitigations

| Challenge | Embree approach | TinyBVH approach |
|---|---|---|
| Analytic shapes (no triangle geometry) | `RTC_GEOMETRY_TYPE_USER` with callbacks | 12-triangle bbox proxies; fallback to `ray_intersect_preliminary_scalar()` on hit |
| LLVM JIT vector widths (1/4/8/16) | Width-specific `rtcIntersect*` | `tinybvh_trace_func_wrapper<..., Width>` template instantiations |
| `PreliminaryIntersection` reconstruction | `geomID` → shape index | `prim_to_shape`/`prim_to_local` + `shapes_registry_ids` DynamicBuffer |
| Double precision | Silent float32 downcast | `BVH_Double` for full double-precision traversal |
| Instance shapes | `rtcSetGeometryTransform()` | TinyBVH TLAS (not yet implemented) |
| Flat shapes (zero-thickness bbox) | N/A (USER geometry) | PROXY_INFLATE = 1e-4 on zero-extent axes |
| Empty scene (no shapes) | N/A | Guard `Intersect()` — `bvhNode` is null until `Build()` called |
| Dangling vertex pointer | N/A | Store vertices in `TinyBVHState::vertices` / `ShapeGroup::m_bvh_vertices` |

---

## Verification

```bash
# Build with TinyBVH
pixi run kernel-configure-tinybvh && pixi run kernel-build

# Run Eradiate plugin tests (from ext/mitsuba/)
pixi run -e test python -m pytest -m "not slow" -p no:robotframework src/eradiate_plugins/tests/
```

Expected: same 4 pre-existing failures as the k-d tree build
(`test_mdistant/test_mpdistant::test_sample_target[*-target_point]`),
no new failures, no segfaults.
