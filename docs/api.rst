API Reference
=============

This page documents the AstralBridge API. AstralBridge is the primary
contribution of this repository; the upstream AION model is included as a
dependency and its reference is provided at the bottom.

.. currentmodule:: astralbridge

Canonical Observation Layer
---------------------------

Survey-neutral observation containers that sit between external survey data and
AION's existing modality classes.

.. automodule:: astralbridge.canonical.observation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: astralbridge.canonical.validators
   :members:
   :undoc-members:
   :show-inheritance:

Adapter Generation and Validation
---------------------------------

Gemma-powered adapter generation, validation, and repair loop.

.. automodule:: astralbridge.adapters.generator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: astralbridge.adapters.validation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: astralbridge.adapters.base
   :members:
   :undoc-members:
   :show-inheritance:

Gemma Client and Prompts
------------------------

Gemma 4 integration using the official ``google-genai`` SDK.

.. automodule:: astralbridge.gemma.client
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: astralbridge.gemma.prompts
   :members:
   :undoc-members:
   :show-inheritance:

AION Integration Bridge
-----------------------

Deterministic mapping from ``CanonicalObservation`` to existing AION modality
objects.

.. automodule:: astralbridge.integration.aion_bridge
   :members:
   :undoc-members:
   :show-inheritance:

Inference Engine
----------------

Real AION inference: encode -> forward -> decode -> decoded prediction.

.. automodule:: astralbridge.inference.engine
   :members:
   :undoc-members:
   :show-inheritance:

Interpreter
-----------

Constrained Gemma interpretation of real AION predictions.

.. automodule:: astralbridge.interpretation.scientific_interpreter
   :members:
   :undoc-members:
   :show-inheritance:

CLI
---

Command-line interface for the full end-to-end pipeline.

.. automodule:: astralbridge.cli
   :members:
   :undoc-members:
   :show-inheritance:

-----------
-----------

Upstream AION Reference
-----------------------

The AION astronomy foundation model (developed by Polymathic AI) is included as
an upstream dependency. The following modules are part of AION, not AstralBridge.

.. currentmodule:: aion

Main Model
~~~~~~~~~~

.. automodule:: aion.model
   :members:
   :undoc-members:
   :show-inheritance:

Modalities
~~~~~~~~~~

.. automodule:: aion.modalities
   :members: Modality, Image, Spectrum, Scalar, LegacySurveyImage, LegacySurveyFluxG, LegacySurveyFluxR, LegacySurveyFluxI, LegacySurveyFluxZ, Z
   :undoc-members:
   :show-inheritance:

Codec Manager
~~~~~~~~~~~~~

.. automodule:: aion.codecs.manager
   :members:
   :undoc-members:
   :show-inheritance:

Codec Configuration
~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.codecs.config
   :members:
   :undoc-members:
   :show-inheritance:
