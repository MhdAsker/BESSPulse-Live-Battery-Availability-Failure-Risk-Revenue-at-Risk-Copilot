"""Controlled, credential-safe ENTSO-E exceptions."""


class EntsoeError(Exception):
    """Base integration error safe to expose to operators."""


class EntsoeAuthenticationError(EntsoeError):
    """Missing or rejected ENTSO-E credential."""


class EntsoeRateLimitError(EntsoeError):
    """Rate limit persisted after bounded retries."""


class EntsoeRequestError(EntsoeError):
    """Permanent request or exhausted transient failure."""


class EntsoeParseError(EntsoeError):
    """Response was not valid supported ENTSO-E XML."""


class EntsoeValidationError(EntsoeError):
    """Input or normalized data violated an explicit contract."""


class EntsoeDataUnavailableError(EntsoeError):
    """ENTSO-E returned no observations for the requested dataset."""
