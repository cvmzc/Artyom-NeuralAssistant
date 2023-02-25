
class LanguageCodeError(Exception):
    """Exception raised when language code is empty or has incorrect value."""


class ImproperlyConfigured(Exception):
    """Exception raised when developer didn't configure the code properly."""


class LanguagePackNotFound(Exception):
    """Exception raised when language pack is not found.

    Exception raised when language pack is not found for the language code
    given."""


class LanguageDetectionError(Exception):
    """Exception raised when language can't be detected.

    Exception raised when language can't be detected for the text given.
    """


class InvalidRegistryItemType(ValueError):
    """Raised when an attempt is made to register an item in the registry.

    Raised when an attempt is made to register an item in the registry which
    does not have a proper type.
    """
