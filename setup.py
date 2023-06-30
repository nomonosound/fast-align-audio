import codecs
import os
import re

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="fast-align-audio",
    version=find_version("fast_align_audio", "__init__.py"),
    description=(
        "A fast python library for aligning similar audio snippets passed in as NumPy arrays."
    ),
    license="ISC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=["fast_align_audio/_fast_align_audio_cffi.py:ffibuilder"],
    install_requires=["cffi>=1.0.0"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
