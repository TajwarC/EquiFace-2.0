from setuptools import setup, find_packages

setup(
    name="equifair",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
        "tqdm",
        "pyyaml",
        "tensorflow",
        "ultralytics",
    ],
    entry_points={
        "console_scripts": [
            "equifair = equifair.__main__:main",
        ],
    },
    author="Your Name",
    description="A package to calculate FPR and FNR for TFLite face verification models",
    license="MIT",
)
