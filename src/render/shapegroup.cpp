#include <mitsuba/core/properties.h>
#include <mitsuba/render/shapegroup.h>
#include <mitsuba/render/optix_api.h>

#if defined(MI_ENABLE_TINYBVH)
// Do NOT define TINYBVH_IMPLEMENTATION here — scene.cpp owns the single
// translation unit that provides it. We only need the declarations.
#  include <tiny_bvh.h>
#  include <mitsuba/render/mesh.h>
#endif

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT ShapeGroup<Float, Spectrum>::ShapeGroup(const Properties &props)
    : Shape<Float, Spectrum>(props) {
    // ID is now stored in base class JitObject

#if !defined(MI_ENABLE_EMBREE) && !defined(MI_ENABLE_TINYBVH)
    if constexpr (!dr::is_cuda_v<Float>)
        m_kdtree = new ShapeKDTree(props);
#endif
    m_shape_types = 0;
    Base::m_shape_type = ShapeType::ShapeGroup;

    // Add children to the underlying data structure
    for (auto &prop : props.objects()) {
        Base *shape = prop.try_get<Base>();
        if (!shape)
            Throw("Tried to add an unsupported object of type \"%s\"", prop.get<ref<Object>>().get());
        if (shape->is_shape_group())
            Throw("Nested ShapeGroup is not permitted");
        if (shape->is_emitter())
            Throw("Instancing of emitters is not supported");
        if (shape->is_instance())
            Throw("Nested instancing is not permitted");
        if (shape->is_sensor())
            Throw("Instancing of sensors is not supported");
        else {
            m_shapes.push_back(shape);
            shape->mark_as_instance();

#if defined(MI_ENABLE_EMBREE) || defined(MI_ENABLE_CUDA) || defined(MI_ENABLE_TINYBVH)
            m_bbox.expand(shape->bbox());
#endif

#if !defined(MI_ENABLE_EMBREE) && !defined(MI_ENABLE_TINYBVH)
            if constexpr (!dr::is_cuda_v<Float>)
                m_kdtree->add_shape(shape);
#endif
            uint32_t type = shape->shape_type();
            m_shape_types |= type;
        }
    }
#if !defined(MI_ENABLE_EMBREE) && !defined(MI_ENABLE_TINYBVH)
    if constexpr (!dr::is_cuda_v<Float>) {
        if (!m_kdtree->ready())
            m_kdtree->build();

        m_bbox = m_kdtree->bbox();
    }
#endif

#if defined(MI_ENABLE_TINYBVH)
    if constexpr (!dr::is_cuda_v<Float>) {
        // Build TinyBVH BLAS over all child shapes.
        // Meshes contribute their actual triangles; analytic shapes get a
        // 12-triangle bounding-box proxy (exact re-intersection on hit).
        if constexpr (std::is_same_v<ScalarFloat, double>) {
            for (size_t ci = 0; ci < m_shapes.size(); ++ci) {
                const Base *shape = m_shapes[ci].get();
                if (shape->is_mesh()) {
                    const Mesh<Float, Spectrum> *mesh =
                        static_cast<const Mesh<Float, Spectrum> *>(shape);
                    const float *vp = mesh->vertex_positions_buffer().data();
                    uint32_t nf     = (uint32_t) mesh->face_count();
                    const uint32_t *fi = mesh->faces_buffer().data();
                    for (uint32_t f = 0; f < nf; ++f) {
                        uint32_t i0=fi[f*3], i1=fi[f*3+1], i2=fi[f*3+2];
                        m_bvh_vertices.push_back({(double)vp[i0*3],(double)vp[i0*3+1],(double)vp[i0*3+2]});
                        m_bvh_vertices.push_back({(double)vp[i1*3],(double)vp[i1*3+1],(double)vp[i1*3+2]});
                        m_bvh_vertices.push_back({(double)vp[i2*3],(double)vp[i2*3+1],(double)vp[i2*3+2]});
                        m_bvh_prim_to_shape.push_back((uint32_t)ci);
                        m_bvh_prim_to_local.push_back(f);
                        m_bvh_prim_is_mesh.push_back(true);
                    }
                } else {
                    // Inflate degenerate (zero-extent) axes so proxy
                    // triangles are non-degenerate for BVH traversal.
                    constexpr double PROXY_INFLATE = 1e-4;
                    ScalarBoundingBox3f box = shape->bbox();
                    double mn[3]={box.min.x(),box.min.y(),box.min.z()};
                    double mx[3]={box.max.x(),box.max.y(),box.max.z()};
                    for (int k=0;k<3;++k)
                        if (mx[k]-mn[k]<PROXY_INFLATE){mn[k]-=PROXY_INFLATE*0.5;mx[k]+=PROXY_INFLATE*0.5;}
                    tinybvh::bvhdbl3 c[8];
                    for (int k=0;k<8;++k)
                        c[k]={(k&1)?mx[0]:mn[0],(k&2)?mx[1]:mn[1],(k&4)?mx[2]:mn[2]};
                    static const int fc[6][4]={{0,1,3,2},{4,6,7,5},{0,4,5,1},{2,3,7,6},{0,2,6,4},{1,5,7,3}};
                    uint32_t local=0;
                    for (int f=0;f<6;++f){
                        m_bvh_vertices.push_back(c[fc[f][0]]);m_bvh_vertices.push_back(c[fc[f][1]]);m_bvh_vertices.push_back(c[fc[f][2]]);
                        m_bvh_prim_to_shape.push_back((uint32_t)ci);m_bvh_prim_to_local.push_back(local++);m_bvh_prim_is_mesh.push_back(false);
                        m_bvh_vertices.push_back(c[fc[f][0]]);m_bvh_vertices.push_back(c[fc[f][2]]);m_bvh_vertices.push_back(c[fc[f][3]]);
                        m_bvh_prim_to_shape.push_back((uint32_t)ci);m_bvh_prim_to_local.push_back(local++);m_bvh_prim_is_mesh.push_back(false);
                    }
                }
            }
            if (!m_bvh_vertices.empty())
                m_tinybvh.Build(m_bvh_vertices.data(), (uint64_t)(m_bvh_vertices.size()/3));
        } else {
            for (size_t ci = 0; ci < m_shapes.size(); ++ci) {
                const Base *shape = m_shapes[ci].get();
                if (shape->is_mesh()) {
                    const Mesh<Float, Spectrum> *mesh =
                        static_cast<const Mesh<Float, Spectrum> *>(shape);
                    const float *vp = mesh->vertex_positions_buffer().data();
                    uint32_t nf     = (uint32_t) mesh->face_count();
                    const uint32_t *fi = mesh->faces_buffer().data();
                    for (uint32_t f = 0; f < nf; ++f) {
                        uint32_t i0=fi[f*3], i1=fi[f*3+1], i2=fi[f*3+2];
                        m_bvh_vertices.push_back({vp[i0*3],vp[i0*3+1],vp[i0*3+2],0.f});
                        m_bvh_vertices.push_back({vp[i1*3],vp[i1*3+1],vp[i1*3+2],0.f});
                        m_bvh_vertices.push_back({vp[i2*3],vp[i2*3+1],vp[i2*3+2],0.f});
                        m_bvh_prim_to_shape.push_back((uint32_t)ci);
                        m_bvh_prim_to_local.push_back(f);
                        m_bvh_prim_is_mesh.push_back(true);
                    }
                } else {
                    constexpr double PROXY_INFLATE = 1e-4;
                    ScalarBoundingBox3f box = shape->bbox();
                    double mn[3]={(double)box.min.x(),(double)box.min.y(),(double)box.min.z()};
                    double mx[3]={(double)box.max.x(),(double)box.max.y(),(double)box.max.z()};
                    for (int k=0;k<3;++k)
                        if (mx[k]-mn[k]<PROXY_INFLATE){mn[k]-=PROXY_INFLATE*0.5;mx[k]+=PROXY_INFLATE*0.5;}
                    tinybvh::bvhvec4 c[8];
                    for (int k=0;k<8;++k)
                        c[k]={(float)((k&1)?mx[0]:mn[0]),(float)((k&2)?mx[1]:mn[1]),(float)((k&4)?mx[2]:mn[2]),0.f};
                    static const int fc[6][4]={{0,1,3,2},{4,6,7,5},{0,4,5,1},{2,3,7,6},{0,2,6,4},{1,5,7,3}};
                    uint32_t local=0;
                    for (int f=0;f<6;++f){
                        m_bvh_vertices.push_back(c[fc[f][0]]);m_bvh_vertices.push_back(c[fc[f][1]]);m_bvh_vertices.push_back(c[fc[f][2]]);
                        m_bvh_prim_to_shape.push_back((uint32_t)ci);m_bvh_prim_to_local.push_back(local++);m_bvh_prim_is_mesh.push_back(false);
                        m_bvh_vertices.push_back(c[fc[f][0]]);m_bvh_vertices.push_back(c[fc[f][2]]);m_bvh_vertices.push_back(c[fc[f][3]]);
                        m_bvh_prim_to_shape.push_back((uint32_t)ci);m_bvh_prim_to_local.push_back(local++);m_bvh_prim_is_mesh.push_back(false);
                    }
                }
            }
            if (!m_bvh_vertices.empty())
                m_tinybvh.Build(m_bvh_vertices.data(), (uint32_t)(m_bvh_vertices.size()/3));
        }
    }
#endif

#if defined(MI_ENABLE_LLVM)
    if constexpr (dr::is_llvm_v<Float>) {
        // Get shapes registry ids
        std::unique_ptr<uint32_t[]> data(new uint32_t[m_shapes.size()]);
        for (size_t i = 0; i < m_shapes.size(); i++)
            data[i] = jit_registry_id(m_shapes[i]);
        m_shapes_registry_ids =
            dr::load<DynamicBuffer<UInt32>>(data.get(), m_shapes.size());
    }
#endif

    // Initialize gradient enabled cache
    m_parameters_grad_enabled_cache = false;
    for (auto s : m_shapes) {
        if (s->parameters_grad_enabled()) {
            m_parameters_grad_enabled_cache = true;
            break;
        }
    }
    m_parameters_grad_enabled_dirty = false;
}

MI_VARIANT ShapeGroup<Float, Spectrum>::~ShapeGroup() {
#if defined(MI_ENABLE_EMBREE)
    if constexpr (!dr::is_cuda_v<Float>) {
        // Ensure all ray tracing kernels are terminated before releasing the scene
        if constexpr (dr::is_llvm_v<Float>)
            dr::sync_thread();

        rtcReleaseScene(m_embree_scene);
    }
#endif
}

MI_VARIANT void ShapeGroup<Float, Spectrum>::traverse(TraversalCallback *cb) {
    for (auto s : m_shapes) {
        std::string_view id = s->id();
        if (id.empty() || string::starts_with(id, "_unnamed_"))
            cb->put("shape", s, ParamFlags::Differentiable);
        else
            cb->put(std::string(id), s, ParamFlags::Differentiable);
    }
}

MI_VARIANT void ShapeGroup<Float, Spectrum>::parameters_changed(const std::vector<std::string> &/*keys*/) {
    for (auto &s : m_shapes) {
        if (s->dirty()) {
            m_dirty = true;
            break;
        }
    }

    // Mark gradient cache as dirty since parameters may have changed
    m_parameters_grad_enabled_dirty = true;

    Base::parameters_changed();
}


MI_VARIANT typename ShapeGroup<Float, Spectrum>::SurfaceInteraction3f
ShapeGroup<Float, Spectrum>::compute_surface_interaction(const Ray3f &ray,
                                                         const PreliminaryIntersection3f &pi,
                                                         uint32_t ray_flags,
                                                         uint32_t recursion_depth,
                                                         Mask active) const {
    MI_MASK_ARGUMENT(active);

    if (recursion_depth > 0)
        return dr::zeros<SurfaceInteraction3f>();

    ShapePtr shape = pi.shape;

    if constexpr (!dr::is_cuda_v<Float>) {
        if constexpr (!dr::is_array_v<Float>) {
            Assert(pi.shape_index < m_shapes.size());
            shape = m_shapes[pi.shape_index];
        } else {
#if defined(MI_ENABLE_LLVM)
            shape = dr::gather<UInt32>(m_shapes_registry_ids, pi.shape_index, active);
#endif
        }
    }

    return shape->compute_surface_interaction(ray, pi, ray_flags, 1, active);
}

MI_VARIANT typename ShapeGroup<Float, Spectrum>::ScalarSize
ShapeGroup<Float, Spectrum>::primitive_count() const {
#if !defined(MI_ENABLE_EMBREE) && !defined(MI_ENABLE_TINYBVH)
    if constexpr (!dr::is_cuda_v<Float>)
        return m_kdtree->primitive_count();
#endif

    ScalarSize count = 0;
    for (auto shape : m_shapes)
        count += shape->primitive_count();

    return count;
}

#if defined(MI_ENABLE_CUDA)
MI_VARIANT void ShapeGroup<Float, Spectrum>::optix_prepare_ias(
    const OptixDeviceContext &context, std::vector<OptixInstance> &instances,
    uint32_t instance_id, const ScalarAffineTransform4f &transf) {
    prepare_ias(context, m_shapes, m_sbt_offset, m_accel, instance_id, transf, instances);
}

MI_VARIANT void ShapeGroup<Float, Spectrum>::optix_fill_hitgroup_records(std::vector<HitGroupSbtRecord> &hitgroup_records,
                                                                         const OptixProgramGroup *pg,
                                                                         const OptixProgramGroupMapping &pg_mapping) {
    m_sbt_offset = (uint32_t) hitgroup_records.size();
    fill_hitgroup_records(m_shapes, hitgroup_records, pg, pg_mapping);
}

MI_VARIANT void ShapeGroup<Float, Spectrum>::optix_prepare_geometry() { }

MI_VARIANT void ShapeGroup<Float, Spectrum>::optix_build_gas(const OptixDeviceContext& context) {
    if (m_dirty) {
        build_gas(context, m_shapes, m_accel);
        for (auto &s : m_shapes)
            s->m_dirty = false;

        m_accel_handles.clear();
        m_accel_handles.push_back(dr::opaque<UInt64>(m_accel.meshes.handle));
        m_accel_handles.push_back(dr::opaque<UInt64>(m_accel.bspline_curves.handle));
        m_accel_handles.push_back(dr::opaque<UInt64>(m_accel.linear_curves.handle));
        m_accel_handles.push_back(dr::opaque<UInt64>(m_accel.custom_shapes.handle));
    }
}
#endif

#if defined(MI_ENABLE_EMBREE)
MI_VARIANT RTCGeometry ShapeGroup<Float, Spectrum>::embree_geometry(RTCDevice device) {
    DRJIT_MARK_USED(device);
    if constexpr (!dr::is_cuda_v<Float>) {
        if (m_dirty) {
            if (m_embree_scene == nullptr)
                m_embree_scene = rtcNewScene(device);

            for (int geo : m_embree_geometries)
                rtcDetachGeometry(m_embree_scene, geo);
            m_embree_geometries.clear();

            for (auto &s : m_shapes) {
                RTCGeometry geom = s->embree_geometry(device);
                m_embree_geometries.push_back(rtcAttachGeometry(m_embree_scene, geom));
                rtcReleaseGeometry(geom);
            }

            // Ensure shape data pointers are finished evaluating before building
            if constexpr (dr::is_llvm_v<Float>)
                dr::sync_thread();

            rtcCommitScene(m_embree_scene);

            for (auto &s : m_shapes)
                s->m_dirty = false;

            // This method is called once per instance, hence make sure we only
            // rebuild the BVH once per update.
            m_dirty = false;
        }

        RTCGeometry instance = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(instance, m_embree_scene);
        return instance;
    } else {
        Throw("embree_geometry() should only be called in CPU mode.");
    }
}
#endif

#if !defined(MI_ENABLE_EMBREE)
MI_VARIANT
std::tuple<typename ShapeGroup<Float, Spectrum>::ScalarFloat,
           typename ShapeGroup<Float, Spectrum>::ScalarPoint2f,
           typename ShapeGroup<Float, Spectrum>::ScalarUInt32,
           typename ShapeGroup<Float, Spectrum>::ScalarUInt32>
ShapeGroup<Float, Spectrum>::ray_intersect_preliminary_scalar(const ScalarRay3f &ray) const {
#if defined(MI_ENABLE_TINYBVH)
    ScalarFloat hit_t   = ray.maxt;
    ScalarPoint2f hit_uv(0.f, 0.f);
    uint32_t hit_shape = (uint32_t) -1;
    uint32_t hit_prim  = (uint32_t) -1;

    if constexpr (std::is_same_v<ScalarFloat, double>) {
        tinybvh::RayEx tbvh_ray(
            tinybvh::bvhdbl3(ray.o.x(), ray.o.y(), ray.o.z()),
            tinybvh::bvhdbl3(ray.d.x(), ray.d.y(), ray.d.z()),
            (double) ray.maxt);
        m_tinybvh.Intersect(tbvh_ray);
        if (tbvh_ray.hit.t < (double) ray.maxt) {
            uint32_t p = (uint32_t) tbvh_ray.hit.prim;
            if (!m_bvh_prim_is_mesh[p]) {
                // Analytic child shape: exact re-intersection
                auto [e_t, e_uv, e_prim, e_inst] =
                    m_shapes[m_bvh_prim_to_shape[p]]->ray_intersect_preliminary_scalar(ray);
                if (e_t < ray.maxt) {
                    hit_t = e_t; hit_uv = e_uv;
                    hit_shape = m_bvh_prim_to_shape[p]; hit_prim = e_prim;
                    (void) e_inst;
                }
            } else {
                hit_t = (ScalarFloat) tbvh_ray.hit.t;
                hit_uv = ScalarPoint2f((ScalarFloat)tbvh_ray.hit.u, (ScalarFloat)tbvh_ray.hit.v);
                hit_shape = m_bvh_prim_to_shape[p];
                hit_prim  = m_bvh_prim_to_local[p];
            }
        }
    } else {
        tinybvh::Ray tbvh_ray(
            tinybvh::bvhvec3((float)ray.o.x(), (float)ray.o.y(), (float)ray.o.z()),
            tinybvh::bvhvec3((float)ray.d.x(), (float)ray.d.y(), (float)ray.d.z()),
            (float) ray.maxt);
        m_tinybvh.Intersect(tbvh_ray);
        if (tbvh_ray.hit.t < (float) ray.maxt) {
            uint32_t p = tbvh_ray.hit.prim;
            if (!m_bvh_prim_is_mesh[p]) {
                auto [e_t, e_uv, e_prim, e_inst] =
                    m_shapes[m_bvh_prim_to_shape[p]]->ray_intersect_preliminary_scalar(ray);
                if (e_t < ray.maxt) {
                    hit_t = e_t; hit_uv = e_uv;
                    hit_shape = m_bvh_prim_to_shape[p]; hit_prim = e_prim;
                    (void) e_inst;
                }
            } else {
                hit_t = (ScalarFloat) tbvh_ray.hit.t;
                hit_uv = ScalarPoint2f((ScalarFloat)tbvh_ray.hit.u, (ScalarFloat)tbvh_ray.hit.v);
                hit_shape = m_bvh_prim_to_shape[p];
                hit_prim  = m_bvh_prim_to_local[p];
            }
        }
    }

    return { hit_t, hit_uv, hit_shape, hit_prim };
#else
    auto pi = m_kdtree->template ray_intersect_scalar<false>(ray);
    return { pi.t, pi.prim_uv, pi.shape_index, pi.prim_index };
#endif
}

MI_VARIANT
bool ShapeGroup<Float, Spectrum>::ray_test_scalar(const ScalarRay3f &ray) const {
#if defined(MI_ENABLE_TINYBVH)
    if constexpr (std::is_same_v<ScalarFloat, double>) {
        tinybvh::RayEx tbvh_ray(
            tinybvh::bvhdbl3(ray.o.x(), ray.o.y(), ray.o.z()),
            tinybvh::bvhdbl3(ray.d.x(), ray.d.y(), ray.d.z()),
            (double) ray.maxt);
        return m_tinybvh.IsOccluded(tbvh_ray);
    } else {
        tinybvh::Ray tbvh_ray(
            tinybvh::bvhvec3((float)ray.o.x(), (float)ray.o.y(), (float)ray.o.z()),
            tinybvh::bvhvec3((float)ray.d.x(), (float)ray.d.y(), (float)ray.d.z()),
            (float) ray.maxt);
        return m_tinybvh.IsOccluded(tbvh_ray);
    }
#else
    return m_kdtree->template ray_intersect_scalar<true>(ray).is_valid();
#endif
}
#endif

MI_VARIANT bool ShapeGroup<Float, Spectrum>::parameters_grad_enabled() const {
    // Recompute cache if dirty
    if (m_parameters_grad_enabled_dirty) {
        m_parameters_grad_enabled_cache = false;
        for (auto s : m_shapes) {
            if (s->parameters_grad_enabled()) {
                m_parameters_grad_enabled_cache = true;
                break;
            }
        }
        m_parameters_grad_enabled_dirty = false;
    }
    return m_parameters_grad_enabled_cache;
}

MI_VARIANT std::string ShapeGroup<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
        oss << "ShapeGroup[" << std::endl
            << "  name = \"" << this->id() << "\"," << std::endl
            << "  prim_count = " << primitive_count() << std::endl
            << "]";
    return oss.str();
}

MI_IMPLEMENT_TRAVERSE_CB(ShapeGroup, Base)
MI_INSTANTIATE_CLASS(ShapeGroup)
NAMESPACE_END(mitsuba)
