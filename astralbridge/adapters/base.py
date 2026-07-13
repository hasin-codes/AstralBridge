"""Base class for generated survey adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from astralbridge.canonical.observation import CanonicalObservation


class SurveyAdapter(ABC):
    """Adapter interface for converting survey data to CanonicalObservation."""

    @abstractmethod
    def convert(self, raw_data: Any) -> CanonicalObservation:
        """Convert survey-specific input into a CanonicalObservation."""
        raise NotImplementedError
