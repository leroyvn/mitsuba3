#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/volume.h>
#include <mitsuba/render/eradiate/extremum.h>

NAMESPACE_BEGIN(mitsuba)

/**!
.. _plugin-volume-irregularcoords:

Irregular coordinate mapping (:monosp:`irregularcoordsvolume`)
--------------------------------------------------------------

.. pluginparameters::

 * - volume
   - |volume|
   - Nested volume plugin whose texture coordinates are to be remapped.
   - —

 * - x, y, z
   - |string|
   - Comma- or space-separated list of strictly increasing node positions (in
     local coordinates) defining the irregular grid along the corresponding
     axis. When an axis is left unspecified, the coordinate is passed through to
     the nested volume unchanged. Each list must contain at least two nodes.
   - —

 * - to_world
   - |transform|
   - Optional 4x4 transformation matrix mapping the local coordinates (in which
     the nodes are expressed) to world coordinates.
   - —

This plugin adapts the texture coordinates of a nested volume so that a
rectilinear (per-axis irregular) grid maps onto the nested volume's uniform
:math:`[0, 1]^3` texture space. Along each specified axis, the node positions
:math:`x_0 < x_1 < \dots < x_{N-1}` are mapped to evenly spaced texture
coordinates :math:`x_i \leftrightarrow i / (N - 1)`, with piecewise-linear
interpolation in between. Coordinates outside the node range are clamped to
:math:`[0, 1]`.

For a nested ``gridvolume``, the number of nodes along an axis must match the
nested resolution along that axis so that node :math:`x_i` aligns with grid
sample :math:`i`.
*/

template <typename Float, typename Spectrum>
class IrregularCoordsVolume final : public Volume<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Volume, m_to_local, m_bbox)
    MI_IMPORT_TYPES(VolumeGrid, ExtremumStructure)

    using VolumeType   = Volume<Float, Spectrum>;
    using FloatStorage = DynamicBuffer<Float>;

    IrregularCoordsVolume(const Properties &props) : Base(props) {
        m_volume = props.get_volume<VolumeType>("volume", 1.f);

        const char *axes[3] = { "x", "y", "z" };
        for (int i = 0; i < 3; ++i)
            m_active[i] = load_axis(props, axes[i], i);

        m_to_local = props.get<ScalarAffineTransform4f>(
                             "to_world", ScalarAffineTransform4f())
                         .inverse();
        update_bbox();
    }

    void add_extremum_structure(ExtremumStructure *extremum) override {
        m_volume->add_extremum_structure(extremum);
    }

    UnpolarizedSpectrum eval(const Interaction3f &it, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        Interaction3f it_mapped = it;
        it_mapped.p             = map_point(it.p, active);
        return m_volume->eval(it_mapped, active);
    }

    Float eval_1(const Interaction3f &it, Mask active = true) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        Interaction3f it_mapped = it;
        it_mapped.p             = map_point(it.p, active);
        return m_volume->eval_1(it_mapped, active);
    }

    ScalarFloat max() const override { return m_volume->max(); }

    ScalarFloat min() const override { return m_volume->min(); }

    ScalarVector3i resolution() const override { return m_volume->resolution(); }

    std::pair<Float, Float> extremum(BoundingBox3f bbox, Mask local) const override {
        if (dr::any(!local))
            NotImplementedError("IrregularCoords only supports local bounds");

        // bbox is expressed in the nested volume's normalized [0,1]^3 space,
        // which is exactly the space this plugin maps into. Forward directly
        // after clamping to the valid range.
        BoundingBox3f clamped(dr::maximum(bbox.min, Point3f(0.f)),
                              dr::minimum(bbox.max, Point3f(1.f)));
        return m_volume->extremum(clamped);
    }

    typename Base::PinGuard pin() const override { return m_volume->pin(); }

    void traverse(TraversalCallback *cb) override {
        cb->put("volume", m_volume.get(), ParamFlags::NonDifferentiable);
        Base::traverse(cb);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "IrregularCoordsVolume[" << std::endl
            << "  to_local = " << string::indent(m_to_local, 13) << "," << std::endl
            << "  bbox = " << string::indent(m_bbox) << "," << std::endl
            << "  volume = " << string::indent(m_volume) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(IrregularCoordsVolume)

private:
    /// Parse an axis node list from the properties; returns true if specified.
    ///
    /// Accepts either a comma/space-separated string or a Dr.Jit array/tensor
    /// (passed from Python, arriving as a ``Type::Any`` wrapping a ``TensorXf``).
    bool load_axis(const Properties &props, const char *name, int axis) {
        if (!props.has_property(name))
            return false;

        FloatStorage buf;
        Properties::Type type = props.type(name);

        if (type == Properties::Type::Any) {
            // Dr.Jit array / tensor input: use the flattened storage directly.
            // Monotonicity is trusted here (checking would force an eval).
            buf = props.get_any<TensorXf>(name).array();
            if (buf.size() < 2)
                Throw("IrregularCoords: axis '%s' needs at least two nodes",
                      name);
        } else if (type == Properties::Type::String) {
            std::vector<std::string> tokens =
                string::tokenize(props.get<std::string>(name), " ,");
            if (tokens.size() < 2)
                Throw("IrregularCoords: axis '%s' needs at least two nodes",
                      name);

            std::vector<ScalarFloat> nodes(tokens.size());
            for (size_t i = 0; i < tokens.size(); ++i) {
                nodes[i] = (ScalarFloat) std::stod(tokens[i]);
                if (i > 0 && nodes[i] <= nodes[i - 1])
                    Throw("IrregularCoords: axis '%s' nodes must be strictly "
                          "increasing", name);
            }
            buf = dr::load<FloatStorage>(nodes.data(), nodes.size());
        } else {
            Throw("IrregularCoords: axis '%s' must be a string or a Dr.Jit "
                  "array", name);
        }

        m_nodes[axis] = buf;
        m_lo[axis]    = dr::slice(buf, 0);
        m_hi[axis]    = dr::slice(buf, buf.size() - 1);
        return true;
    }

    /// Map a world-space point to the nested volume's normalized coordinates.
    Point3f map_point(const Point3f &p_world, Mask active) const {
        Point3f p = m_to_local * p_world;
        for (int i = 0; i < 3; ++i)
            if (m_active[i])
                p[i] = remap_axis(p[i], i, active);
        return p;
    }

    /// Piecewise-linear remap of a local coordinate to [0, 1] along one axis.
    Float remap_axis(Float w, int axis, Mask active) const {
        const FloatStorage &nodes = m_nodes[axis];
        uint32_t n                = (uint32_t) nodes.size();

        UInt32 idx = dr::binary_search<UInt32>(
            0, n, [&](UInt32 i) DRJIT_INLINE_LAMBDA {
                return dr::gather<Float>(nodes, i, active) < w;
            });
        idx = dr::maximum(dr::minimum(idx, n - 1u), 1u) - 1u;

        Float x0 = dr::gather<Float>(nodes, idx, active),
              x1 = dr::gather<Float>(nodes, idx + 1u, active);
        Float alpha = (w - x0) / (x1 - x0);
        Float t     = (Float(idx) + alpha) * (ScalarFloat(1) / ScalarFloat(n - 1));
        return dr::clip(t, 0.f, 1.f);
    }

    void update_bbox() {
        // Local extent: node range on specified axes, [0, 1] otherwise.
        ScalarPoint3f lo(m_active[0] ? m_lo[0] : 0.f,
                         m_active[1] ? m_lo[1] : 0.f,
                         m_active[2] ? m_lo[2] : 0.f),
                      hi(m_active[0] ? m_hi[0] : 1.f,
                         m_active[1] ? m_hi[1] : 1.f,
                         m_active[2] ? m_hi[2] : 1.f);

        ScalarAffineTransform4f to_world = m_to_local.inverse();
        m_bbox = ScalarBoundingBox3f();
        for (int c = 0; c < 8; ++c) {
            ScalarPoint3f corner((c & 1) ? hi.x() : lo.x(),
                                 (c & 2) ? hi.y() : lo.y(),
                                 (c & 4) ? hi.z() : lo.z());
            m_bbox.expand(to_world * corner);
        }
    }

    ref<VolumeType> m_volume;
    FloatStorage m_nodes[3];
    bool m_active[3]   = { false, false, false };
    ScalarPoint3f m_lo = ScalarPoint3f(0.f);
    ScalarPoint3f m_hi = ScalarPoint3f(1.f);

    MI_TRAVERSE_CB(Base, m_volume)
};

MI_EXPORT_PLUGIN(IrregularCoordsVolume)

NAMESPACE_END(mitsuba)
