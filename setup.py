#!/usr/bin/env python

import os
from setuptools import setup
import sys

requires = open('requirements.txt').read().strip().split('\n')
install_requires = []
extras_require = {}
for r in requires:
    if ';' in r:
        # requirements.txt conditional dependencies need to be reformatted for wheels
        # to the form: `'[extra_name]:condition' : ['requirements']`
        req, cond = r.split(';', 1)
        cond = ':' + cond
        cond_reqs = extras_require.setdefault(cond, [])
        cond_reqs.append(req)
    else:
        install_requires.append(r)

setup(name='wxsbi',
      version='0.0.1',
      description='A flexible toolkit for stochastic weather simulation using numpyro/jax and simulation-based inference.',
      author='Brian Groenke, Jakob Wessel',
      maintainer='Brian Groenke',
      maintainer_email='brian.groenke@awi.de',
      license='MIT',
      install_requires=install_requires,
      extras_require=extras_require,
      packages=['weathergen','wxsbi'],
      long_description=(open('README.md').read() if os.path.exists('README.md')
                        else ''),
      zip_safe=False)
