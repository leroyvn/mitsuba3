#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _spectrum-discrete:

Discrete spectrum (:monosp:`discrete`)
----------------------------------------

This spectrum returns a constant value at fixed wavelengths.

TODO:
* Sample from multiple wavelengths

*/

template <typename Float, typename Spectrum>
class DiscreteSpectrum final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    using FloatStorage = DynamicBuffer<Float>;

    DiscreteSpectrum(const Properties &props) : Texture(props) {
        // Collect wavelengths
        std::vector<std::string> wavelengths_str =
            string::tokenize(props.string("wavelengths"), " ,");
        std::vector<ScalarFloat> wavelengths_data;
        wavelengths_data.reserve(wavelengths_str.size());

        for (const auto &s : wavelengths_str) {
            try {
                wavelengths_data.push_back(string::stof<ScalarFloat>(s));
            } catch (...) {
                Throw("While parsing wavelengths: could not parse floating "
                      "point value '%s'",
                      s);
            }
        }
        m_wavelengths = dr::load<FloatStorage>(wavelengths_data.data(),
                                               wavelengths_data.size());

        // Collect weights (if undefined, use uniform distribution)
        FloatStorage weights;

        if (props.has_property("weights")) {
            std::vector<std::string> weights_str =
                string::tokenize(props.string("weights"), " ,");

            if (weights_str.size() != wavelengths_data.size()) {
                Throw("'weights' and 'wavelengths' arrays must have the same "
                      "size");
            }

            std::vector<ScalarFloat> weights_data;
            weights_data.reserve(weights_str.size());

            for (const auto &s : weights_str) {
                try {
                    weights_data.push_back(string::stof<ScalarFloat>(s));
                } catch (...) {
                    Throw("While parsing weights: could not parse floating "
                          "point value '%s'",
                          s);
                }
            }
            weights = dr::load<FloatStorage>(weights_data.data(),
                                             weights_data.size());
        } else {
            weights = dr::full<FloatStorage>(1.f, wavelengths_data.size());
        }

        m_distr =
            DiscreteDistribution<Wavelength>(weights.data(), weights.size());
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("wavelengths", m_wavelengths,
                                +ParamFlags::NonDifferentiable);
        callback->put_parameter("weights", m_distr.pmf(),
                                +ParamFlags::NonDifferentiable);
    }

    void
    parameters_changed(const std::vector<std::string> & /*keys*/) override {
        m_distr.update();
        if (m_distr.pmf().size() != m_wavelengths.size()) {
            Throw("'weights' and 'wavelengths' arrays must have the same "
                  "size");
        }
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f & /*si*/,
                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        // This is a Dirac spectrum: always evaluate to 0
        return depolarizer<Spectrum>(0.f);
    }

    Wavelength pdf_spectrum(const SurfaceInteraction3f & /*si*/,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        // This is a Dirac spectrum: always evaluate to 0
        return Wavelength(0.f);
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample_spectrum(const SurfaceInteraction3f & /*si*/,
                    const Wavelength &sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureSample, active);

        if constexpr (is_spectral_v<Spectrum>) {
            auto [indexes, weights] = m_distr.sample_pmf(sample, active);
            return { dr::gather<Wavelength>(m_wavelengths, indexes), weights };
        } else {
            DRJIT_MARK_USED(sample);
            NotImplementedError("sample");
        }
    }

    Float mean() const override { NotImplementedError("mean"); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DiscreteSpectrum[" << std::endl
            << "  wavelengths = " << string::indent(m_wavelengths) << ","
            << std::endl
            << "  distr = " << string::indent(m_distr) << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

private:
    DiscreteDistribution<Wavelength> m_distr;
    FloatStorage m_wavelengths;
};

MI_IMPLEMENT_CLASS_VARIANT(DiscreteSpectrum, Texture)
MI_EXPORT_PLUGIN(DiscreteSpectrum, "Singleton spectrum")
NAMESPACE_END(mitsuba)
