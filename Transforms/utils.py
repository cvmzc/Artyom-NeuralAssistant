# -*- coding: utf-8 -*-
from collections import Counter
import re
import unicodedata
from base import registry
from conf import get_setting
from discover import autodiscover
from exceptions import (
    LanguageCodeError,
    LanguagePackNotFound,
)


def splitbyx(n, x, format_int=True):
    length = len(n)
    if length > x:
        start = length % x
        if start > 0:
            result = n[:start]
            yield int(result) if format_int else result
        for i in range(start, length, x):
            result = n[i:i+x]
            yield int(result) if format_int else result
    else:
        yield int(n) if format_int else n


def get_digits(n):
    a = [int(x) for x in reversed(list(('%03d' % n)[-3:]))]
    return a

def ensure_autodiscover():
    """Ensure autodiscover."""
    # Running autodiscover if registry is empty
    if not registry.registry:
        autodiscover()


def get_translit_function(language_code):
    """Return translit function for the language given.

    :param str language_code:
    :return callable:
    """
    ensure_autodiscover()

    cls = registry.get(language_code)
    if cls is None:
        raise LanguagePackNotFound(
            ("Language pack for code %s is not found." % language_code)
        )

    language_pack = cls()
    return language_pack.translit


def translit(value, language_code=None, reversed=False, strict=False):
    ensure_autodiscover()

    cls = registry.get(language_code)

    if cls is None:
        raise LanguagePackNotFound(
            ("Language pack for code %s is not found." % language_code)
        )

    language_pack = cls()
    return language_pack.translit(value, reversed=reversed, strict=strict)


def suggest(value, language_code=None, reversed=False, limit=None):
    """Suggest possible variants.

    :param str value:
    :param str language_code:
    :param bool reversed: If set to True, reversed translation is made.
    :param int limit: Limit number of suggested variants.
    :return list:
    """
    ensure_autodiscover()

    if language_code is None and reversed is False:
        raise LanguageCodeError(
            ("``language_code`` is optional with ``reversed`` set to True "
              "only.")
        )

    cls = registry.get(language_code)

    if cls is None:
        raise LanguagePackNotFound(
            ("Language pack for code %s is not found." % language_code)
        )

    language_pack = cls()

    return language_pack.suggest(value, reversed=reversed, limit=limit)


def get_language_pack(language_code):
    """Get registered language pack by language code given.

    :param str language_code:
    :return transliterate.base.TranslitLanguagePack: Returns None on failure.
    """
    ensure_autodiscover()
    return registry.registry.get(language_code, None)


# Strips numbers from unicode string.
def strip_numbers(text):
    """Strip numbers from text."""
    return ''.join(filter(lambda u: not u.isdigit(), text))


def extract_most_common_words(text, num_words=None):
    """Extract most common words.

    :param unicode text:
    :param int num_words:
    :return list:
    """
    if num_words is None:
        num_words = get_setting('LANGUAGE_DETECTION_MAX_NUM_KEYWORDS')

    text = strip_numbers(text)
    counter = Counter()
    for word in text.split(' '):
        if len(word) > 1:
            counter[word] += 1
    return counter.most_common(num_words)


def slugify(text, language_code=None):
    if language_code:
        transliterated_text = translit(text, language_code, reversed=True)
        slug = unicodedata.normalize('NFKD', transliterated_text) \
                          .encode('ascii', 'ignore') \
                          .decode('ascii')
        slug = re.sub(r'[^\w\s-]', '', slug).strip().lower()
        return re.sub(r'[-\s]+', '-', slug)