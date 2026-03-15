// TinyBVH backend for Mitsuba's CPU acceleration structure.
// Mirrors the structure of scene_native.inl (k-d tree) and scene_embree.inl.
//
// This file is included from scene.cpp when MI_ENABLE_TINYBVH is defined.
// TinyBVH is a header-only library; we instantiate its implementation here.
#define TINYBVH_IMPLEMENTATION
#include <tiny_bvh.h>

#include <mitsuba/render/mesh.h>

NAMESPACE_BEGIN(mitsuba)

// ---------------------------------------------------------------------------
// RayHitT — SOA buffer layout used by jit_llvm_ray_trace
// ---------------------------------------------------------------------------

#if defined(_MSC_VER)
#  pragma pack(push, 1)
#endif

template <typename ScalarFloat> struct RayHitT {
    ScalarFloat o_x, o_y, o_z, tnear;
    ScalarFloat d_x, d_y, d_z, time;
    ScalarFloat tfar;
    uint32_t mask, id, flags;
    ScalarFloat ng_x, ng_y, ng_z, u, v;
    uint32_t prim_id, geom_id, inst_id;
} DRJIT_PACK;

#if defined(_MSC_VER)
#  pragma pack(pop)
#endif

// ---------------------------------------------------------------------------
// State struct
// ---------------------------------------------------------------------------

template <typename Float, typename Spectrum>
struct TinyBVHState {
    MI_IMPORT_CORE_TYPES()
    using ShapeT  = Shape<Float, Spectrum>;
    using MeshT   = Mesh<Float, Spectrum>;

    // Select BVH type based on scalar float precision.
    // Double-precision variants → BVH_Double; single → BVH.
    using ScalarF = ScalarFloat;
    using BVHType = std::conditional_t<std::is_same_v<ScalarF, double>,
                                       tinybvh::BVH_Double,
                                       tinybvh::BVH>;

    BVHType bvh;

    // Vertex data that backs the BVH (TinyBVH stores a pointer, not a copy).
    // Must outlive the BVH object.
    using VertexType = std::conditional_t<std::is_same_v<ScalarF, double>,
                                          tinybvh::bvhdbl3,
                                          tinybvh::bvhvec4>;
    std::vector<VertexType> vertices;

    // JIT-managed shape pointer table — identical to EmbreeState/NativeState.
    DynamicBuffer<UInt32> shapes_registry_ids;

    // Maps BVH primitive index → index into Scene::m_shapes
    std::vector<uint32_t> prim_to_shape;
    // Maps BVH primitive index → primitive-local index within that shape
    std::vector<uint32_t> prim_to_local;
    // true if the BVH prim comes from a mesh; false = bbox proxy of analytic shape
    std::vector<bool> prim_is_mesh;

    // Flat array of raw shape pointers for use inside the trace wrapper.
    // Populated from Scene::m_shapes in accel_parameters_changed_cpu.
    std::vector<const ShapeT *> shapes;

    // Function pointer registered with the JIT (LLVM path only)
    void *func_ptr = nullptr;
    UInt64 func_handle;
};

// ---------------------------------------------------------------------------
// Forward declaration of the JIT trace wrapper
// ---------------------------------------------------------------------------

template <typename Float, typename Spectrum, bool ShadowRay, size_t Width>
void tinybvh_trace_func_wrapper(const int *valid, void *ptr,
                                void * /* context */, uint8_t *args);

// ---------------------------------------------------------------------------
// accel_init_cpu
// ---------------------------------------------------------------------------

MI_VARIANT void Scene<Float, Spectrum>::accel_init_cpu(const Properties &/*props*/) {
    using State = TinyBVHState<Float, Spectrum>;

    m_accel = new State();
    State &s = *(State *) m_accel;

    if constexpr (dr::is_llvm_v<Float>) {
        if (!m_shapes.empty()) {
            std::unique_ptr<uint32_t[]> data(new uint32_t[m_shapes.size()]);
            for (size_t i = 0; i < m_shapes.size(); i++)
                data[i] = jit_registry_id(m_shapes[i]);
            s.shapes_registry_ids =
                dr::load<DynamicBuffer<UInt32>>(data.get(), m_shapes.size());
        } else {
            s.shapes_registry_ids = dr::zeros<DynamicBuffer<UInt32>>();
        }
    }

    accel_parameters_changed_cpu();
}

// ---------------------------------------------------------------------------
// accel_parameters_changed_cpu
// ---------------------------------------------------------------------------

MI_VARIANT void Scene<Float, Spectrum>::accel_parameters_changed_cpu() {
    using State  = TinyBVHState<Float, Spectrum>;
    using ScalarF = typename State::ScalarF;
    using MeshT   = typename State::MeshT;

    if constexpr (dr::is_llvm_v<Float>)
        dr::sync_thread();

    State &s = *(State *) m_accel;
    s.prim_to_shape.clear();
    s.prim_to_local.clear();
    s.prim_is_mesh.clear();
    // Clear vertex storage — BVH stores a pointer into this buffer,
    // so it must be populated before Build() and kept alive afterward.
    s.vertices.clear();

    // Populate the flat shape pointer list used in the trace wrapper.
    s.shapes.clear();
    for (auto &sh : m_shapes)
        s.shapes.push_back(sh.get());

    ScopedPhase phase(ProfilerPhase::InitAccel);

    // Epsilon used to inflate zero-extent bbox dimensions for analytic shapes.
    // Flat shapes (e.g. disk, arectangle) have a zero-thickness bounding box in
    // one axis. Without inflation the 12-triangle proxy degenerates: all 8 box
    // corners collapse, producing zero-area triangles that TinyBVH cannot
    // traverse. The proxy only needs to be geometrically non-degenerate; any
    // false-positive hit is filtered out by the exact re-intersection step.
    // The value is chosen to be well above floating-point noise (1e-4 world units
    // for the default scale) while being negligible compared to any real shape.
    constexpr double PROXY_INFLATE = 1e-4;

    // Build proxy triangles for a bounding box, inflating zero-extent axes.
    // VT is either tinybvh::bvhvec4 (float) or tinybvh::bvhdbl3 (double).
    auto append_bbox_proxy_triangles = [&](auto &vertices, const ScalarBoundingBox3f &box,
                                           uint32_t shape_idx, auto make_vert) {
        using VT = std::remove_reference_t<decltype(vertices[0])>;
        double mn[3] = { (double)box.min.x(), (double)box.min.y(), (double)box.min.z() };
        double mx[3] = { (double)box.max.x(), (double)box.max.y(), (double)box.max.z() };
        // Inflate degenerate (zero-extent) axes so proxy triangles are non-planar.
        for (int k = 0; k < 3; ++k) {
            if (mx[k] - mn[k] < PROXY_INFLATE) {
                mn[k] -= PROXY_INFLATE * 0.5;
                mx[k] += PROXY_INFLATE * 0.5;
            }
        }
        VT c[8];
        for (int k = 0; k < 8; ++k)
            c[k] = make_vert((k & 1) ? mx[0] : mn[0],
                             (k & 2) ? mx[1] : mn[1],
                             (k & 4) ? mx[2] : mn[2]);
        static const int fc[6][4] = {
            {0,1,3,2},{4,6,7,5},{0,4,5,1},{2,3,7,6},{0,2,6,4},{1,5,7,3}
        };
        uint32_t local = 0;
        for (int f = 0; f < 6; ++f) {
            vertices.push_back(c[fc[f][0]]); vertices.push_back(c[fc[f][1]]); vertices.push_back(c[fc[f][2]]);
            s.prim_to_shape.push_back(shape_idx); s.prim_to_local.push_back(local++); s.prim_is_mesh.push_back(false);
            vertices.push_back(c[fc[f][0]]); vertices.push_back(c[fc[f][2]]); vertices.push_back(c[fc[f][3]]);
            s.prim_to_shape.push_back(shape_idx); s.prim_to_local.push_back(local++); s.prim_is_mesh.push_back(false);
        }
    };

    if constexpr (std::is_same_v<ScalarF, double>) {
        for (size_t i = 0; i < m_shapes.size(); ++i) {
            const Shape *shape = m_shapes[i].get();
            if (shape->is_mesh()) {
                const MeshT *mesh = static_cast<const MeshT *>(shape);
                dr::sync_thread();
                const float *vp = mesh->vertex_positions_buffer().data();
                uint32_t nf = (uint32_t) mesh->face_count();
                const uint32_t *fi = mesh->faces_buffer().data();
                for (uint32_t f = 0; f < nf; ++f) {
                    uint32_t i0 = fi[f*3+0], i1 = fi[f*3+1], i2 = fi[f*3+2];
                    s.vertices.push_back(tinybvh::bvhdbl3((double)vp[i0*3], (double)vp[i0*3+1], (double)vp[i0*3+2]));
                    s.vertices.push_back(tinybvh::bvhdbl3((double)vp[i1*3], (double)vp[i1*3+1], (double)vp[i1*3+2]));
                    s.vertices.push_back(tinybvh::bvhdbl3((double)vp[i2*3], (double)vp[i2*3+1], (double)vp[i2*3+2]));
                    s.prim_to_shape.push_back((uint32_t) i);
                    s.prim_to_local.push_back(f);
                    s.prim_is_mesh.push_back(true);
                }
            } else {
                append_bbox_proxy_triangles(s.vertices, shape->bbox(), (uint32_t)i,
                    [](double x, double y, double z) { return tinybvh::bvhdbl3(x, y, z); });
            }
        }
        if (!s.vertices.empty())
            s.bvh.Build(s.vertices.data(), (uint64_t)(s.vertices.size() / 3));

    } else {
        for (size_t i = 0; i < m_shapes.size(); ++i) {
            const Shape *shape = m_shapes[i].get();
            if (shape->is_mesh()) {
                const MeshT *mesh = static_cast<const MeshT *>(shape);
                dr::sync_thread();
                const float *vp = mesh->vertex_positions_buffer().data();
                uint32_t nf = (uint32_t) mesh->face_count();
                const uint32_t *fi = mesh->faces_buffer().data();
                for (uint32_t f = 0; f < nf; ++f) {
                    uint32_t i0 = fi[f*3+0], i1 = fi[f*3+1], i2 = fi[f*3+2];
                    s.vertices.push_back(tinybvh::bvhvec4(vp[i0*3], vp[i0*3+1], vp[i0*3+2], 0.f));
                    s.vertices.push_back(tinybvh::bvhvec4(vp[i1*3], vp[i1*3+1], vp[i1*3+2], 0.f));
                    s.vertices.push_back(tinybvh::bvhvec4(vp[i2*3], vp[i2*3+1], vp[i2*3+2], 0.f));
                    s.prim_to_shape.push_back((uint32_t) i);
                    s.prim_to_local.push_back(f);
                    s.prim_is_mesh.push_back(true);
                }
            } else {
                append_bbox_proxy_triangles(s.vertices, shape->bbox(), (uint32_t)i,
                    [](double x, double y, double z) { return tinybvh::bvhvec4((float)x, (float)y, (float)z, 0.f); });
            }
        }
        if (!s.vertices.empty())
            s.bvh.Build(s.vertices.data(), (uint32_t)(s.vertices.size() / 3));
    }

    // Set up JIT callback to release the state when the handle is freed.
    if constexpr (dr::is_llvm_v<Float>) {
        if (m_accel_handle.index())
            jit_var_set_callback(m_accel_handle.index(), nullptr, nullptr);
        m_accel_handle = UInt64::map_(m_accel, 1, false);
        jit_var_set_callback(
            m_accel_handle.index(),
            [](uint32_t, int free, void *payload) {
                if (free) {
                    jit_enqueue_host_func(
                        JitBackend::LLVM,
                        [](void *p) {
                            Log(Debug, "Free TinyBVH state..");
                            delete (State *) p;
                        },
                        payload
                    );
                }
            },
            (void *) m_accel
        );

        int jit_width = jit_llvm_vector_width();
        void *func_ptr = nullptr;
        switch (jit_width) {
            case 1:  func_ptr = (void *) tinybvh_trace_func_wrapper<Float, Spectrum, false, 1>;  break;
            case 4:  func_ptr = (void *) tinybvh_trace_func_wrapper<Float, Spectrum, false, 4>;  break;
            case 8:  func_ptr = (void *) tinybvh_trace_func_wrapper<Float, Spectrum, false, 8>;  break;
            case 16: func_ptr = (void *) tinybvh_trace_func_wrapper<Float, Spectrum, false, 16>; break;
            default:
                Throw("accel_parameters_changed_cpu(): Dr.Jit is configured "
                      "for vectors of width %u, which is not supported by "
                      "the TinyBVH backend!", jit_width);
        }
        s.func_ptr    = func_ptr;
        s.func_handle = UInt64::map_(func_ptr, 1, false);
    }

    clear_shapes_dirty();
}

// ---------------------------------------------------------------------------
// accel_release_cpu
// ---------------------------------------------------------------------------

MI_VARIANT void Scene<Float, Spectrum>::accel_release_cpu() {
    if constexpr (dr::is_llvm_v<Float>) {
        dr::sync_thread();
        m_accel_handle = 0;
        m_accel = nullptr;
    } else {
        delete (TinyBVHState<Float, Spectrum> *) m_accel;
        m_accel = nullptr;
    }
}

// ---------------------------------------------------------------------------
// tinybvh_trace_func_wrapper
// Mirrors kdtree_trace_func_wrapper in scene_native.inl.
// Called from the LLVM JIT kernel for each SIMD lane.
// ---------------------------------------------------------------------------

template <typename Float, typename Spectrum, bool ShadowRay, size_t Width>
void tinybvh_trace_func_wrapper(const int *valid, void *ptr,
                                void * /* context */, uint8_t *args) {
    MI_IMPORT_TYPES()
    using ScalarRay3f = Ray<ScalarPoint3f, Spectrum>;
    using State  = TinyBVHState<Float, Spectrum>;
    using ScalarF = typename State::ScalarF;
    using RayHit  = RayHitT<ScalarF>;

    State *s = (State *) ptr;

    // Empty scene: BVH was not built; nothing to intersect.
    if (s->vertices.empty())
        return;

    for (size_t i = 0; i < Width; ++i) {
        if (valid[i] == 0)
            continue;

        ScalarF ox = ((ScalarF *) &args[offsetof(RayHit, o_x)  * Width])[i];
        ScalarF oy = ((ScalarF *) &args[offsetof(RayHit, o_y)  * Width])[i];
        ScalarF oz = ((ScalarF *) &args[offsetof(RayHit, o_z)  * Width])[i];
        ScalarF dx = ((ScalarF *) &args[offsetof(RayHit, d_x)  * Width])[i];
        ScalarF dy = ((ScalarF *) &args[offsetof(RayHit, d_y)  * Width])[i];
        ScalarF dz = ((ScalarF *) &args[offsetof(RayHit, d_z)  * Width])[i];

        ScalarF &ray_maxt = ((ScalarF *) &args[offsetof(RayHit, tfar) * Width])[i];
        ScalarF  ray_time = ((ScalarF *) &args[offsetof(RayHit, time) * Width])[i];

        // Query the BVH (full intersection; we check prim_is_mesh afterward).
        uint32_t hit_prim = (uint32_t) -1;
        ScalarF  hit_t    = ray_maxt;
        ScalarF  hit_u    = 0, hit_v = 0;

        if constexpr (std::is_same_v<ScalarF, double>) {
            tinybvh::RayEx ray(tinybvh::bvhdbl3(ox, oy, oz),
                               tinybvh::bvhdbl3(dx, dy, dz),
                               (double) ray_maxt);
            s->bvh.Intersect(ray);
            if (ray.hit.t < (double) ray_maxt) {
                hit_prim = (uint32_t) ray.hit.prim;
                hit_t    = (ScalarF) ray.hit.t;
                hit_u    = (ScalarF) ray.hit.u;
                hit_v    = (ScalarF) ray.hit.v;
            }
        } else {
            tinybvh::Ray ray(tinybvh::bvhvec3((float)ox, (float)oy, (float)oz),
                             tinybvh::bvhvec3((float)dx, (float)dy, (float)dz),
                             (float) ray_maxt);
            s->bvh.Intersect(ray);
            if (ray.hit.t < (float) ray_maxt) {
                hit_prim = ray.hit.prim;
                hit_t    = (ScalarF) ray.hit.t;
                hit_u    = (ScalarF) ray.hit.u;
                hit_v    = (ScalarF) ray.hit.v;
            }
        }

        if (hit_prim == (uint32_t) -1)
            continue;

        uint32_t shape_idx = s->prim_to_shape[hit_prim];
        bool     is_mesh   = s->prim_is_mesh[hit_prim];

        if (!is_mesh) {
            // Analytic shape: the BVH hit a bbox proxy triangle.
            // Re-intersect exactly using the shape's own routine.
            ScalarRay3f exact_ray(
                ScalarPoint3f(ox, oy, oz),
                ScalarVector3f(dx, dy, dz),
                (ScalarFloat) ray_maxt, (ScalarFloat) ray_time,
                wavelength_t<Spectrum>());
            auto [e_t, e_uv, e_prim, e_inst] =
                s->shapes[shape_idx]->ray_intersect_preliminary_scalar(exact_ray);
            if (e_t >= (ScalarFloat) ray_maxt)
                continue;

            if constexpr (ShadowRay) {
                ray_maxt = ScalarF(0);
            } else {
                ScalarF &ref_tfar  = ((ScalarF *)  &args[offsetof(RayHit, tfar)    * Width])[i];
                ScalarF &ref_u     = ((ScalarF *)  &args[offsetof(RayHit, u)       * Width])[i];
                ScalarF &ref_v     = ((ScalarF *)  &args[offsetof(RayHit, v)       * Width])[i];
                uint32_t &ref_prim = ((uint32_t *) &args[offsetof(RayHit, prim_id) * Width])[i];
                uint32_t &ref_geom = ((uint32_t *) &args[offsetof(RayHit, geom_id) * Width])[i];
                uint32_t &ref_inst = ((uint32_t *) &args[offsetof(RayHit, inst_id) * Width])[i];
                ref_tfar = (ScalarF) e_t;
                ref_u    = (ScalarF) e_uv[0];
                ref_v    = (ScalarF) e_uv[1];
                ref_prim = e_prim;
                ref_geom = shape_idx;
                ref_inst = (uint32_t) -1;
                (void) e_inst;
            }
        } else {
            if constexpr (ShadowRay) {
                ray_maxt = ScalarF(0);
            } else {
                ScalarF &ref_tfar  = ((ScalarF *)  &args[offsetof(RayHit, tfar)    * Width])[i];
                ScalarF &ref_u     = ((ScalarF *)  &args[offsetof(RayHit, u)       * Width])[i];
                ScalarF &ref_v     = ((ScalarF *)  &args[offsetof(RayHit, v)       * Width])[i];
                uint32_t &ref_prim = ((uint32_t *) &args[offsetof(RayHit, prim_id) * Width])[i];
                uint32_t &ref_geom = ((uint32_t *) &args[offsetof(RayHit, geom_id) * Width])[i];
                uint32_t &ref_inst = ((uint32_t *) &args[offsetof(RayHit, inst_id) * Width])[i];
                ref_tfar = hit_t;
                ref_u    = hit_u;
                ref_v    = hit_v;
                ref_prim = s->prim_to_local[hit_prim];
                ref_geom = shape_idx;
                ref_inst = (uint32_t) -1;
            }
        }
        (void) ray_time;
    }
}

// ---------------------------------------------------------------------------
// ray_intersect_preliminary_cpu
// ---------------------------------------------------------------------------

MI_VARIANT typename Scene<Float, Spectrum>::PreliminaryIntersection3f
Scene<Float, Spectrum>::ray_intersect_preliminary_cpu(const Ray3f &ray,
                                                      Mask coherent,
                                                      Mask active) const {
    using State  = TinyBVHState<Float, Spectrum>;
    using ScalarF = typename State::ScalarF;
    using ScalarRay3f = Ray<ScalarPoint3f, Spectrum>;
    State &s = *(State *) m_accel;

    if constexpr (!dr::is_array_v<Float>) {
        DRJIT_MARK_USED(coherent);
        PreliminaryIntersection3f pi = dr::zeros<PreliminaryIntersection3f>();

        // Guard: an empty scene has no geometry and no built BVH.
        // Calling Intersect on an unbuilt BVH dereferences a null bvhNode ptr.
        if (s.vertices.empty())
            return pi;

        if constexpr (std::is_same_v<ScalarF, double>) {
            tinybvh::RayEx tbvh_ray(
                tinybvh::bvhdbl3((double)ray.o.x(), (double)ray.o.y(), (double)ray.o.z()),
                tinybvh::bvhdbl3((double)ray.d.x(), (double)ray.d.y(), (double)ray.d.z()),
                (double) ray.maxt);
            s.bvh.Intersect(tbvh_ray);
            if (tbvh_ray.hit.t < (double) ray.maxt) {
                uint32_t prim      = (uint32_t) tbvh_ray.hit.prim;
                uint32_t shape_idx = s.prim_to_shape[prim];
                uint32_t local_idx = s.prim_to_local[prim];
                const Shape *shape = m_shapes[shape_idx].get();
                if (!shape->is_mesh()) {
                    ScalarRay3f exact_ray(
                        ScalarPoint3f((ScalarFloat)ray.o.x(), (ScalarFloat)ray.o.y(), (ScalarFloat)ray.o.z()),
                        ScalarVector3f((ScalarFloat)ray.d.x(), (ScalarFloat)ray.d.y(), (ScalarFloat)ray.d.z()),
                        (ScalarFloat)ray.maxt, (ScalarFloat)ray.time, wavelength_t<Spectrum>());
                    auto [e_t, e_uv, e_prim, e_inst] =
                        shape->ray_intersect_preliminary_scalar(exact_ray);
                    if (e_t < (ScalarFloat)ray.maxt) {
                        pi.t           = (Float) e_t;
                        pi.prim_uv     = Point2f((Float)e_uv[0], (Float)e_uv[1]);
                        pi.prim_index  = e_prim;
                        pi.shape_index = shape_idx;
                        pi.shape       = shape;
                        (void) e_inst;
                    }
                } else {
                    pi.t           = (Float) tbvh_ray.hit.t;
                    pi.prim_uv     = Point2f((Float)tbvh_ray.hit.u, (Float)tbvh_ray.hit.v);
                    pi.prim_index  = local_idx;
                    pi.shape_index = shape_idx;
                    pi.shape       = shape;
                }
            }
        } else {
            tinybvh::Ray tbvh_ray(
                tinybvh::bvhvec3((float)ray.o.x(), (float)ray.o.y(), (float)ray.o.z()),
                tinybvh::bvhvec3((float)ray.d.x(), (float)ray.d.y(), (float)ray.d.z()),
                (float) ray.maxt);
            s.bvh.Intersect(tbvh_ray);
            if (tbvh_ray.hit.t < (float) ray.maxt) {
                uint32_t prim      = tbvh_ray.hit.prim;
                uint32_t shape_idx = s.prim_to_shape[prim];
                uint32_t local_idx = s.prim_to_local[prim];
                const Shape *shape = m_shapes[shape_idx].get();
                if (!shape->is_mesh()) {
                    ScalarRay3f exact_ray(
                        ScalarPoint3f((ScalarFloat)ray.o.x(), (ScalarFloat)ray.o.y(), (ScalarFloat)ray.o.z()),
                        ScalarVector3f((ScalarFloat)ray.d.x(), (ScalarFloat)ray.d.y(), (ScalarFloat)ray.d.z()),
                        (ScalarFloat)ray.maxt, (ScalarFloat)ray.time, wavelength_t<Spectrum>());
                    auto [e_t, e_uv, e_prim, e_inst] =
                        shape->ray_intersect_preliminary_scalar(exact_ray);
                    if (e_t < (ScalarFloat)ray.maxt) {
                        pi.t           = (Float) e_t;
                        pi.prim_uv     = Point2f((Float)e_uv[0], (Float)e_uv[1]);
                        pi.prim_index  = e_prim;
                        pi.shape_index = shape_idx;
                        pi.shape       = shape;
                        (void) e_inst;
                    }
                } else {
                    pi.t           = (Float) tbvh_ray.hit.t;
                    pi.prim_uv     = Point2f((Float)tbvh_ray.hit.u, (Float)tbvh_ray.hit.v);
                    pi.prim_index  = local_idx;
                    pi.shape_index = shape_idx;
                    pi.shape       = shape;
                }
            }
        }

        return pi;
    } else {
        // LLVM JIT path — identical to scene_native.inl.
        void *func_ptr  = s.func_ptr,
             *scene_ptr = m_accel;

        UInt64 func_v = UInt64::steal(
                   jit_var_pointer(JitBackend::LLVM, func_ptr, s.func_handle.index(), 0)),
               scene_v = UInt64::steal(
                   jit_var_pointer(JitBackend::LLVM, scene_ptr, m_accel_handle.index(), 0));

        UInt32 zero = dr::zeros<UInt32>();
        Float ray_mint = dr::zeros<Float>();

        uint32_t in[14] = { coherent.index(),  active.index(),
                            ray.o.x().index(), ray.o.y().index(),
                            ray.o.z().index(), ray_mint.index(),
                            ray.d.x().index(), ray.d.y().index(),
                            ray.d.z().index(), ray.time.index(),
                            ray.maxt.index(),  zero.index(),
                            zero.index(),      zero.index() };
        uint32_t out[6] { };

        jit_llvm_ray_trace(func_v.index(), scene_v.index(), 0, in, out);

        PreliminaryIntersection3f pi;
        Float t(Float::steal(out[0]));
        pi.prim_uv     = Vector2f(Float::steal(out[1]), Float::steal(out[2]));
        pi.prim_index  = UInt32::steal(out[3]);
        pi.shape_index = UInt32::steal(out[4]);

        UInt32 inst_index = UInt32::steal(out[5]);
        Mask hit      = active && (t != ray.maxt);
        pi.t          = dr::select(hit, t, dr::Infinity<Float>);

        Mask hit_inst = hit && (inst_index != ((uint32_t) -1));
        UInt32 index  = dr::select(hit_inst, inst_index, pi.shape_index);
        ShapePtr shape = dr::gather<UInt32>(s.shapes_registry_ids, index, hit);
        pi.instance = shape & hit_inst;
        pi.shape    = shape & !hit_inst;

        return pi;
    }
}

// ---------------------------------------------------------------------------
// ray_intersect_cpu
// ---------------------------------------------------------------------------

MI_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect_cpu(const Ray3f &ray, uint32_t ray_flags,
                                          Mask coherent, Mask active) const {
    if constexpr (!dr::is_cuda_v<Float>) {
        PreliminaryIntersection3f pi =
            ray_intersect_preliminary_cpu(ray, coherent, active);
        return pi.compute_surface_interaction(ray, ray_flags, active);
    } else {
        DRJIT_MARK_USED(ray); DRJIT_MARK_USED(ray_flags);
        DRJIT_MARK_USED(coherent); DRJIT_MARK_USED(active);
        Throw("ray_intersect_cpu() should only be called in CPU mode.");
    }
}

// ---------------------------------------------------------------------------
// ray_test_cpu
// ---------------------------------------------------------------------------

MI_VARIANT typename Scene<Float, Spectrum>::Mask
Scene<Float, Spectrum>::ray_test_cpu(const Ray3f &ray,
                                     Mask coherent, Mask active) const {
    using State = TinyBVHState<Float, Spectrum>;
    State &s = *(State *) m_accel;

    if constexpr (!dr::is_jit_v<Float>) {
        // Scalar path: reuse full intersection (handles analytic shapes).
        DRJIT_MARK_USED(coherent); DRJIT_MARK_USED(s);
        PreliminaryIntersection3f pi =
            ray_intersect_preliminary_cpu(ray, Mask(true), Mask(true));
        return Mask(pi.is_valid());
    } else {
        // LLVM JIT path: same func_ptr as intersection, shadow flag = 1.
        void *func_ptr  = s.func_ptr,
             *scene_ptr = m_accel;

        UInt64 func_v = UInt64::steal(
                   jit_var_pointer(JitBackend::LLVM, func_ptr, s.func_handle.index(), 0)),
               scene_v = UInt64::steal(
                   jit_var_pointer(JitBackend::LLVM, scene_ptr, m_accel_handle.index(), 0));

        UInt32 zero = dr::zeros<UInt32>();
        Float ray_mint = dr::zeros<Float>();

        uint32_t in[14] = { coherent.index(),  active.index(),
                            ray.o.x().index(), ray.o.y().index(),
                            ray.o.z().index(), ray_mint.index(),
                            ray.d.x().index(), ray.d.y().index(),
                            ray.d.z().index(), ray.time.index(),
                            ray.maxt.index(),  zero.index(),
                            zero.index(),      zero.index() };
        uint32_t out[1] { };

        jit_llvm_ray_trace(func_v.index(), scene_v.index(), 1, in, out);

        return active && (Float::steal(out[0]) != ray.maxt);
    }
}

// ---------------------------------------------------------------------------
// ray_intersect_naive_cpu
// ---------------------------------------------------------------------------

MI_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect_naive_cpu(const Ray3f &ray,
                                                Mask active) const {
    PreliminaryIntersection3f pi =
        ray_intersect_preliminary_cpu(ray, Mask(false), active);
    return pi.compute_surface_interaction(ray, +RayFlags::All, active);
}

// ---------------------------------------------------------------------------
// Traversal callbacks (frozen function support)
// ---------------------------------------------------------------------------

MI_VARIANT void Scene<Float, Spectrum>::traverse_1_cb_ro_cpu(
    void *payload, drjit::detail::traverse_callback_ro fn) const {
    if constexpr (dr::is_llvm_v<Float>) {
        TinyBVHState<Float, Spectrum> &s =
            *(TinyBVHState<Float, Spectrum> *) m_accel;
        drjit::traverse_1_fn_ro(s.shapes_registry_ids, payload, fn);
        drjit::traverse_1_fn_ro(s.func_handle, payload, fn);
    }
}

MI_VARIANT void Scene<Float, Spectrum>::traverse_1_cb_rw_cpu(
    void *payload, drjit::detail::traverse_callback_rw fn) {
    if constexpr (dr::is_llvm_v<Float>) {
        TinyBVHState<Float, Spectrum> &s =
            *(TinyBVHState<Float, Spectrum> *) m_accel;
        drjit::traverse_1_fn_rw(s.shapes_registry_ids, payload, fn);
        drjit::traverse_1_fn_rw(s.func_handle, payload, fn);
    }
}

NAMESPACE_END(mitsuba)
