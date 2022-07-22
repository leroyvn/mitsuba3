import drjit as dr
import pytest

import mitsuba as mi


def test_construct(variant_scalar_rgb):
    # Minimal form: weights are all equal to 1
    s = mi.load_dict(
        {
            "type": "discrete",
            "wavelengths": "300, 400, 500, 600",
        }
    )
    params = mi.traverse(s)
    assert len(params["wavelengths"]) == len(params["weights"])
    assert dr.allclose(params["weights"], 1.0)

    # Assigning weights: incorrect weight array size raises
    with pytest.raises(RuntimeError):
        mi.load_dict(
            {
                "type": "discrete",
                "wavelengths": "300, 400, 500, 600",
                "weights": "",
            }
        )

    # Assigning weights: appropriate values are set
    s = mi.load_dict(
        {
            "type": "discrete",
            "wavelengths": "300, 400, 500, 600",
            "weights": "1, 2, 2, 1",
        }
    )
    params = mi.traverse(s)
    assert len(params["wavelengths"]) == len(params["weights"])
    assert dr.allclose(params["weights"], [1.0, 2.0, 2.0, 1.0])


def test_traverse(variant_scalar_rgb):
    s = mi.load_dict(
        {
            "type": "discrete",
            "wavelengths": "300, 400, 500, 600",
        }
    )
    assert "pmf = [1, 1, 1, 1]" in str(s)
    params = mi.traverse(s)

    # Updating weights and wavelengths has the expected effect
    params["weights"] = [1, 2, 2, 1]
    params["wavelengths"] = [200, 300, 400, 500]
    params.update()
    assert "pmf = [1, 2, 2, 1]" in str(s)
    assert "wavelengths = [200, 300, 400, 500]" in str(s)

    # Setting inappropriate sizes results in an exception
    params["wavelengths"] = [200, 300, 400]
    with pytest.raises(RuntimeError):
        params.update()


def test_eval(variant_scalar_spectral):
    s = mi.load_dict(
        {
            "type": "discrete",
            "wavelengths": "300, 400, 500, 600",
        }
    )
    si = mi.SurfaceInteraction3f()
    for i in range(5):
        si.wavelengths = 450 + 50 * i
        assert dr.allclose(s.eval(si), 0.0)
        assert dr.allclose(s.pdf_spectrum(si), 0.0)

    with pytest.raises(RuntimeError) as excinfo:
        s.eval_1(si)
    assert "not implemented" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        s.eval_3(si)
    assert "not implemented" in str(excinfo.value)


def test_sample(variant_scalar_spectral):
    s = mi.load_dict(
        {
            "type": "discrete",
            "wavelengths": "300, 400, 500, 600",
            "weights": "1, 2, 2, 1",
        }
    )
    si = mi.SurfaceInteraction3f()

    for (sample, expected_wavelength, expected_weight) in [
        (0.0, 300.0, 1 / 6),
        (0.25, 400.0, 1 / 3),
        (0.5, 400.0, 1 / 3),
        (0.75, 500, 1 / 3),
        (1.0, 600, 1 / 6),
    ]:
        wavelengths, weights = s.sample_spectrum(si, sample)
        assert dr.allclose(wavelengths, expected_wavelength)
        assert dr.allclose(weights, expected_weight)
