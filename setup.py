from setuptools import setup, find_packages

setup(
    name="equiface",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
        "tqdm",
        "pyyaml",
        "tensorflow",
        "ultralytics"
        "deepface",
    ],
    entry_points={
        "console_scripts": [
            "equiface = equiface.__main__:main",
        ],
    },
    author="Tajwar Choudhury",
    description="A package to calculate FPR and FNR for TFLite face verification models",
    license="MIT",
)
