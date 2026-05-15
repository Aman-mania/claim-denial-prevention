"""Week 7 remediation-agent layer.

Import concrete classes from their modules to avoid loading optional heavy
runtime dependencies during lightweight tests.
"""

__all__ = ["RemediationAgent", "OpenAIPresentationLayer", "RecommendationCatalog"]
