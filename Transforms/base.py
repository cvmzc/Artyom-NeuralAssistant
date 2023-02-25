# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
from collections import OrderedDict
from decimal import Decimal

from compat import to_s
from currency import parse_currency_parts, prefix_currency
import six

from exceptions import (
    ImproperlyConfigured,
    InvalidRegistryItemType
)

class FilteringTransforms_Base(object):
    CURRENCY_FORMS = {}
    CURRENCY_ADJECTIVES = {}

    def __init__(self):
        self.is_title = False
        self.precision = 2
        self.exclude_title = []
        self.negword = "(-) "
        self.pointword = "(.)"
        self.errmsg_nonnum = "type(%s) not in [long, int, float]"
        self.errmsg_floatord = "Cannot treat float %s as ordinal."
        self.errmsg_negord = "Cannot treat negative num %s as ordinal."
        self.errmsg_toobig = "abs(%s) must be less than %s."

        self.setup()

        # uses cards
        if any(hasattr(self, field) for field in
               ['high_numwords', 'mid_numwords', 'low_numwords']):
            self.cards = OrderedDict()
            self.set_numwords()
            self.MAXVAL = 1000 * list(self.cards.keys())[0]

    def set_numwords(self):
        self.set_high_numwords(self.high_numwords)
        self.set_mid_numwords(self.mid_numwords)
        self.set_low_numwords(self.low_numwords)

    def set_high_numwords(self, *args):
        raise NotImplementedError

    def set_mid_numwords(self, mid):
        for key, val in mid:
            self.cards[key] = val

    def set_low_numwords(self, numwords):
        for word, n in zip(numwords, range(len(numwords) - 1, -1, -1)):
            self.cards[n] = word

    def splitnum(self, value):
        for elem in self.cards:
            if elem > value:
                continue

            out = []
            if value == 0:
                div, mod = 1, 0
            else:
                div, mod = divmod(value, elem)

            if div == 1:
                out.append((self.cards[1], 1))
            else:
                if div == value:  # The system tallies, eg Roman Numerals
                    return [(div * self.cards[elem], div*elem)]
                out.append(self.splitnum(div))

            out.append((self.cards[elem], elem))

            if mod:
                out.append(self.splitnum(mod))

            return out

    def parse_minus(self, num_str):
        """Detach minus and return it as symbol with new num_str."""
        if num_str.startswith('-'):
            # Extra spacing to compensate if there is no minus.
            return '%s ' % self.negword.strip(), num_str[1:]
        return '', num_str

    def str_to_number(self, value):
        return Decimal(value)

    def to_cardinal(self, value):
        try:
            assert int(value) == value
        except (ValueError, TypeError, AssertionError):
            return self.to_cardinal_float(value)

        out = ""
        if value < 0:
            value = abs(value)
            out = "%s " % self.negword.strip()

        if value >= self.MAXVAL:
            raise OverflowError(self.errmsg_toobig % (value, self.MAXVAL))

        val = self.splitnum(value)
        words, num = self.clean(val)
        return self.title(out + words)

    def float2tuple(self, value):
        pre = int(value)

        # Simple way of finding decimal places to update the precision
        self.precision = abs(Decimal(str(value)).as_tuple().exponent)

        post = abs(value - pre) * 10**self.precision
        if abs(round(post) - post) < 0.01:
            # We generally floor all values beyond our precision (rather than
            # rounding), but in cases where we have something like 1.239999999,
            # which is probably due to python's handling of floats, we actually
            # want to consider it as 1.24 instead of 1.23
            post = int(round(post))
        else:
            post = int(math.floor(post))

        return pre, post

    def to_cardinal_float(self, value):
        try:
            float(value) == value
        except (ValueError, TypeError, AssertionError, AttributeError):
            raise TypeError(self.errmsg_nonnum % value)

        pre, post = self.float2tuple(float(value))

        post = str(post)
        post = '0' * (self.precision - len(post)) + post

        out = [self.to_cardinal(pre)]
        if self.precision:
            out.append(self.title(self.pointword))

        for i in range(self.precision):
            curr = int(post[i])
            out.append(to_s(self.to_cardinal(curr)))

        return " ".join(out)

    def merge(self, curr, next):
        raise NotImplementedError

    def clean(self, val):
        out = val
        while len(val) != 1:
            out = []
            left, right = val[:2]
            if isinstance(left, tuple) and isinstance(right, tuple):
                out.append(self.merge(left, right))
                if val[2:]:
                    out.append(val[2:])
            else:
                for elem in val:
                    if isinstance(elem, list):
                        if len(elem) == 1:
                            out.append(elem[0])
                        else:
                            out.append(self.clean(elem))
                    else:
                        out.append(elem)
            val = out
        return out[0]

    def title(self, value):
        if self.is_title:
            out = []
            value = value.split()
            for word in value:
                if word in self.exclude_title:
                    out.append(word)
                else:
                    out.append(word[0].upper() + word[1:])
            value = " ".join(out)
        return value

    def verify_ordinal(self, value):
        if not value == int(value):
            raise TypeError(self.errmsg_floatord % value)
        if not abs(value) == value:
            raise TypeError(self.errmsg_negord % value)

    def to_ordinal(self, value):
        return self.to_cardinal(value)

    def to_ordinal_num(self, value):
        return value

    # Trivial version
    def inflect(self, value, text):
        text = text.split("/")
        if value == 1:
            return text[0]
        return "".join(text)

    # //CHECK: generalise? Any others like pounds/shillings/pence?
    def to_splitnum(self, val, hightxt="", lowtxt="", jointxt="",
                    divisor=100, longval=True, cents=True):
        out = []

        if isinstance(val, float):
            high, low = self.float2tuple(val)
        else:
            try:
                high, low = val
            except TypeError:
                high, low = divmod(val, divisor)

        if high:
            hightxt = self.title(self.inflect(high, hightxt))
            out.append(self.to_cardinal(high))
            if low:
                if longval:
                    if hightxt:
                        out.append(hightxt)
                    if jointxt:
                        out.append(self.title(jointxt))
            elif hightxt:
                out.append(hightxt)

        if low:
            if cents:
                out.append(self.to_cardinal(low))
            else:
                out.append("%02d" % low)
            if lowtxt and longval:
                out.append(self.title(self.inflect(low, lowtxt)))

        return " ".join(out)

    def to_year(self, value, **kwargs):
        return self.to_cardinal(value)

    def pluralize(self, n, forms):
        """
        Should resolve gettext form:
        http://docs.translatehouse.org/projects/localization-guide/en/latest/l10n/pluralforms.html
        """
        raise NotImplementedError

    def _money_verbose(self, number, currency):
        return self.to_cardinal(number)

    def _cents_verbose(self, number, currency):
        return self.to_cardinal(number)

    def _cents_terse(self, number, currency):
        return "%02d" % number

    def to_currency(self, val, currency='EUR', cents=True, separator=',',
                    adjective=False):
        """
        Args:
            val: Numeric value
            currency (str): Currency code
            cents (bool): Verbose cents
            separator (str): Cent separator
            adjective (bool): Prefix currency name with adjective
        Returns:
            str: Formatted string
        """
        left, right, is_negative = parse_currency_parts(val)

        try:
            cr1, cr2 = self.CURRENCY_FORMS[currency]

        except KeyError:
            raise NotImplementedError(
                'Currency code "%s" not implemented for "%s"' %
                (currency, self.__class__.__name__))

        if adjective and currency in self.CURRENCY_ADJECTIVES:
            cr1 = prefix_currency(self.CURRENCY_ADJECTIVES[currency], cr1)

        minus_str = "%s " % self.negword.strip() if is_negative else ""
        money_str = self._money_verbose(left, currency)
        cents_str = self._cents_verbose(right, currency) \
            if cents else self._cents_terse(right, currency)

        return u'%s%s %s%s %s %s' % (
            minus_str,
            money_str,
            self.pluralize(left, cr1),
            separator,
            cents_str,
            self.pluralize(right, cr2)
        )

    def setup(self):
        pass

class TranslitLanguagePack(object):

    language_code = None
    language_name = None
    character_ranges = None
    mapping = None
    reversed_specific_mapping = None

    reversed_pre_processor_mapping = None  # Added
    reversed_pre_processor_mapping_keys = []

    reversed_specific_pre_processor_mapping = None
    reversed_specific_pre_processor_mapping_keys = []

    pre_processor_mapping = None
    pre_processor_mapping_keys = []

    detectable = False
    characters = None
    reversed_characters = None

    def __init__(self):
        try:
            assert self.language_code is not None
            assert self.language_name is not None
            assert self.mapping
        except AssertionError:
            raise ImproperlyConfigured(
                "You should define ``language_code``, ``language_name`` and "
                "``mapping`` properties in your subclassed "
                "``TranslitLanguagePack`` class."
            )

        super(TranslitLanguagePack, self).__init__()

        # Creating a translation table from the mapping set.
        self.translation_table = {}

        for key, val in zip(*self.mapping):
            self.translation_table.update({ord(key): ord(val)})

        # Creating a reversed translation table.
        self.reversed_translation_table = dict(
            zip(self.translation_table.values(), self.translation_table.keys())
        )

        # If any pre-processor rules defined, reversing them for later use.
        if self.pre_processor_mapping:
            self.pre_processor_mapping_keys = self.pre_processor_mapping.keys()
            # If no `reversed_pre_processor_mapping` is defined, construct
            # from `pre_processor_mapping`.
            if not self.reversed_pre_processor_mapping:
                self.reversed_pre_processor_mapping = dict(
                    zip(
                        self.pre_processor_mapping.values(),
                        self.pre_processor_mapping.keys()
                    )
                )
            self.reversed_pre_processor_mapping_keys = \
                self.reversed_pre_processor_mapping.keys()

        else:
            self.reversed_pre_processor_mapping = None

        if self.reversed_specific_mapping:
            self.reversed_specific_translation_table = {}
            for key, val in zip(*self.reversed_specific_mapping):
                self.reversed_specific_translation_table.update(
                    {ord(key): ord(val)}
                )

        if self.reversed_specific_pre_processor_mapping:
            self.reversed_specific_pre_processor_mapping_keys = \
                self.reversed_specific_pre_processor_mapping.keys()

        self._characters = '[^]'

        if self.characters is not None:
            self._characters = '[^{0}]'.format(
                '\\'.join(list(self.characters))
            )

        self._reversed_characters = '[^]'
        if self.reversed_characters is not None:
            self._reversed_characters = \
                '[^{0}]'.format('\\'.join(list(self.characters)))

    def translit(self, value, reversed=False, strict=False,
                 fail_silently=True):
        """Transliterate the given value according to the rules.

        Rules are set in the transliteration pack.

        :param str value:
        :param bool reversed:
        :param bool strict:
        :param bool fail_silently:
        :return str:
        """
        if not six.PY3:
            value = str(value)

        if reversed:
            # Handling reversed specific translations (one side only).
            if self.reversed_specific_mapping:
                value = value.translate(
                    self.reversed_specific_translation_table
                )

            if self.reversed_specific_pre_processor_mapping:
                for rule in self.reversed_specific_pre_processor_mapping_keys:
                    value = value.replace(
                        rule,
                        self.reversed_specific_pre_processor_mapping[rule]
                    )

            # Handling pre-processor mappings.
            if self.reversed_pre_processor_mapping:
                for rule in self.reversed_pre_processor_mapping_keys:
                    value = value.replace(
                        rule,
                        self.reversed_pre_processor_mapping[rule]
                    )

            return value.translate(self.reversed_translation_table)

        if self.pre_processor_mapping:
            for rule in self.pre_processor_mapping_keys:
                value = value.replace(rule, self.pre_processor_mapping[rule])
        res = value.translate(self.translation_table)

        if strict:
            res = self._make_strict(value=res,
                                    reversed=reversed,
                                    fail_silently=fail_silently)

        return res



class TranslitRegistry(object):
    """Language pack registry."""

    def __init__(self):
        self._registry = {}
        self._forced = []

    @property
    def registry(self):
        """Registry."""
        return self._registry

    def register(self, cls, force=False):
        """Register the language pack in the registry.

        :param transliterate.base.LanguagePack cls: Subclass of
            ``transliterate.base.LanguagePack``.
        :param bool force: If set to True, item stays forced. It's not possible
            to un-register a forced item.
        :return bool: True if registered and False otherwise.
        """
        if not issubclass(cls, TranslitLanguagePack):
            raise InvalidRegistryItemType(
                "Invalid item type `%s` for registry `%s`",
                cls,
                self.__class__
            )

        # If item has not been forced yet, add/replace its' value in the
        # registry.
        if force:

            if cls.language_code not in self._forced:
                self._registry[cls.language_code] = cls
                self._forced.append(cls.language_code)
                return True
            else:
                return False

        else:

            if cls.language_code in self._registry:
                return False
            else:
                self._registry[cls.language_code] = cls
                return True

    def unregister(self, cls):
        """Un-registers an item from registry.

        :param transliterate.base.LanguagePack cls: Subclass of
            ``transliterate.base.LanguagePack``.
        :return bool: True if unregistered and False otherwise.
        """
        if not issubclass(cls, TranslitLanguagePack):
            raise InvalidRegistryItemType(
                "Invalid item type `%s` for registry `%s`",
                cls,
                self.__class__
            )

        # Only non-forced items are allowed to be unregistered.
        if cls.language_code in self._registry \
                and cls.language_code not in self._forced:

            self._registry.pop(cls.language_code)
            return True
        else:
            return False

    def get(self, language_code, default=None):
        """Get the given language pack from the registry.

        :param str language_code:
        :return transliterate.base.LanguagePack: Subclass of
            ``transliterate.base.LanguagePack``.
        """
        return self._registry.get(language_code, default)


# Register languages by calling registry.register()
registry = TranslitRegistry()