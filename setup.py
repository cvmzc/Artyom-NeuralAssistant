# coding: utf-8

from cx_Freeze import setup, Executable

executables = [Executable('Artyom.py')]

excludes = ['unicodedata', 'urllib','torch','silero','pyqt5','xml', 'bz2']

options = {
    'build_exe': {
        'include_msvcr': True,
        'excludes': excludes,
    }
}

setup(name='Artyom',
      version='0.0.1',
      description='Artyom - neural assistant',
      executables=executables,
      options=options)