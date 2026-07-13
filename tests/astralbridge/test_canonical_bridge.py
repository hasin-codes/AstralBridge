import pytest
import torch

from aion.modalities import LegacySurveyFluxG, LegacySurveyImage
from astralbridge import CanonicalImage, CanonicalObservation, CanonicalScalar
from astralbridge.integration import AionBridgeError, canonical_to_aion


def test_canonical_to_aion_maps_supported_modalities():
    observation = CanonicalObservation(
        survey="Roman WFI demo",
        images=[
            CanonicalImage(
                data=torch.zeros(1, 4, 96, 96),
                bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
            )
        ],
        scalars=[CanonicalScalar("flux_g", 2.5, "nanomaggie", "WFI_G_FLUX")],
    )

    modalities = canonical_to_aion(observation)

    assert isinstance(modalities[0], LegacySurveyImage)
    assert isinstance(modalities[1], LegacySurveyFluxG)
    assert modalities[1].value.shape == (1, 1)


def test_canonical_to_aion_rejects_new_untrained_bands():
    observation = CanonicalObservation(
        survey="Roman WFI demo",
        images=[CanonicalImage(data=torch.zeros(1, 1, 96, 96), bands=["ROMAN-F146"])],
    )

    with pytest.raises(AionBridgeError):
        canonical_to_aion(observation)
