# -*- coding: utf-8 -*-
import os
from six import print_

try:
    from importlib import import_module
except ImportError:
    import_module = __import__

from conf import get_setting
import sys

try:
    PY32 = (sys.version_info[0] == 3 and sys.version_info[1] == 2)
except Exception as err:
    PY32 = False

def project_dir(base):
    """Project dir."""
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            (os.path.join(*base) if isinstance(base, (list, tuple)) else base)
        ).replace('\\', '/')
    )


PROJECT_DIR = project_dir

def autodiscover():
    """Auto-discover the language packs in contrib/apps."""
    languages_dir = get_setting('LANGUAGES_DIR')
    language_pack_module_name = get_setting('LANGUAGE_PACK_MODULE_NAME')
    debug = get_setting('DEBUG')

    for app_path in os.listdir(PROJECT_DIR(languages_dir)):
        full_app_path = list(languages_dir)
        full_app_path.append(app_path)
        if os.path.isdir(PROJECT_DIR(full_app_path)):
            try:
                import_module(
                    "transliterate.{0}.{1}.{2}".format(
                        '.'.join(languages_dir),
                        app_path,
                        language_pack_module_name
                    )
                )
            except ImportError as err:
                if debug:
                    print_(err)
            except Exception as err:
                if debug:
                    print_(err)
        else:
            pass
