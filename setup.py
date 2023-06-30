from setuptools import setup, find_packages

setup(
    name="fast-align-audio",
    version="0.1",
    packages=find_packages(),
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=["fast_align_audio/_fast_align_audio_cffi.py:ffibuilder"],
    install_requires=["cffi>=1.0.0"],
)
