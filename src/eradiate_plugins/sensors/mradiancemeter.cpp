#include <drjit/tensor.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _plugin-sensor-mradiancemeter:

Multi-radiance meter (:monosp:`mradiancemeter`)
-----------------------------------------------

.. pluginparameters::

 * - origins
   - |string|
   - Comma separated list of locations from which the sensors will be recording
     in world coordinates.
   - —

 * - directions
   - |string|
   - Comma separated list of directions in which the sensors are pointing in
     world coordinates.
   - —

This sensor plugin implements multiple radiance meters, as implemented in the
:monosp:`radiancemeter` plugin.

This sensor allows using the inherent parallelization of Mitsuba2, which is not
possible with the :monosp:`radiancemeter` due to its film size of 1x1.

The origin points and direction vectors for this sensor are specified as a list
of floating point values, where three subsequent values will be grouped into a
point or vector respectively.

The following snippet shows how to specify a :monosp:`mradiancemeter` with
two sensors, one located at (1, 0, 0) and pointing in the direction (-1, 0, 0),
the other located at (0, 1, 0) and pointing in the direction (0, -1, 0).

.. code-block:: xml

    <sensor version="2.0.0" type="mradiancemeter">
        <string name="origins" value="1, 0, 0, 0, 1, 0"/>
        <string name="directions" value="-1, 0, 0, 0, -1, 0"/>
        <film type="hdrfilm">
            <integer name="width" value="2"/>
            <integer name="height" value="1"/>
            <rfilter type="box"/>
        </film>
    </sensor>

.. code-block:: xml
    :name: sphere-meter

    <shape type="sphere">
        <sensor type="irradiancemeter">
            <!-- film -->
        </sensor>
    </shape>

*/

MI_VARIANT class MultiRadianceMeterSensor final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_film, m_to_world, m_needs_sample_2,
                    m_needs_sample_3, sample_wavelengths)
    MI_IMPORT_TYPES()

    using Matrix = dr::Matrix<Float, AffineTransform4f::Size>;
    using Index  = dr::uint32_array_t<Float>;

    MultiRadianceMeterSensor(const Properties &props) : Base(props) {
        // Collect directions and set transforms accordingly
        if (props.has_property("to_world")) {
            Throw("This sensor is specified through a set of origin and "
                  "direction values and cannot use the to_world transform.");
        }

        std::vector<std::string> origins_str =
            string::tokenize(props.get<std::string>("origins"), " ,");
        std::vector<std::string> directions_str =
            string::tokenize(props.get<std::string>("directions"), " ,");

        if (origins_str.size() % 3 != 0)
            Throw("Invalid specification! Number of parameters %s, is not a "
                  "multiple of three.",
                  origins_str.size());

        if (origins_str.size() != directions_str.size())
            Throw("Invalid specification! Number of parameters for origins and "
                  "directions (%s, %s) "
                  "are not equal.",
                  origins_str.size(), directions_str.size());

        m_sensor_count = (size_t) origins_str.size() / 3.f;
        size_t width   = m_sensor_count * 16;
        std::vector<ScalarFloat> buffer(width);

        // TODO: update transform storage loading with new TensorXf interface
        for (size_t i = 0; i < m_sensor_count; ++i) {
            size_t index = i * 3;
            ScalarPoint3f origin =
                ScalarPoint3f(std::stof(origins_str[index]),
                              std::stof(origins_str[index + 1]),
                              std::stof(origins_str[index + 2]));

            ScalarVector3f direction =
                ScalarVector3f(std::stof(directions_str[index]),
                               std::stof(directions_str[index + 1]),
                               std::stof(directions_str[index + 2]));

            ScalarPoint3f target = origin + direction;
            auto [up, _]         = coordinate_system(direction);
            ScalarAffineTransform4f transform =
                ScalarAffineTransform4f::look_at(origin, target, up).matrix;
            memcpy(&buffer[i * 16], &transform, sizeof(ScalarFloat) * 16);
        }

        size_t shape[3] = { (size_t) m_sensor_count, 4, 4 };
        m_transforms    = TensorXf(buffer.data(), 3, shape);

        // Check film size
        if (!dr::all(m_film->size() == ScalarPoint2i(m_transforms.size() / 16, 1)))
            Throw("Film size must be [n_radiancemeters, 1]. Expected %s, "
                  "found: %s",
                  ScalarPoint2i(m_transforms.size() / 16, 1), m_film->size());

        // Check reconstruction filter radius
        if (m_film->rfilter()->radius() >
            0.5f + math::RayEpsilon<Float>)
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. default box)");

        m_needs_sample_2 = true;
        m_needs_sample_3 = false;
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f & /*aperture_sample*/,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        Ray3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);
        ray.wavelengths = wavelengths;

        // 2. Set ray origin and direction
        // Clamp to valid range: sample_ray_differential adds a pixel offset
        // (+1/width) to compute differentials, which can push position_sample.x()
        // above 1.0 for the last pixel, causing an out-of-bounds gather.
        Int32 sensor_index = dr::clip(
            Int32(position_sample.x() * m_sensor_count),
            0, Int32(m_sensor_count - 1));
        Index index(sensor_index);

        Matrix coefficients =
            dr::gather<Matrix>(m_transforms.array(), index, active);
        AffineTransform4f trafo(coefficients);
        ray.o = trafo * Point3f{ 0.f, 0.f, 0.f };
        ray.d = trafo * Vector3f{ 0.f, 0.f, 1.f };

        return { ray, wav_weight };
    }

    ScalarBoundingBox3f bbox() const override {
        // Return an invalid bounding box
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MultiRadianceMeterSensor[" << std::endl
            << "  transforms = " << m_transforms.array() << "," << std::endl
            << "  film = " << m_film << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(MultiRadianceMeterSensor)
private:
    TensorXf m_transforms;
    size_t m_sensor_count;
};

MI_EXPORT_PLUGIN(MultiRadianceMeterSensor)

NAMESPACE_END(mitsuba)
