"""
Setup configuration for uq_benchmarks package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="uq_benchmarks",
    version="0.1.0",
    author="UQ Research Team",
    description="Uncertainty quantification benchmarking with formal disentanglement methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "keras": [
            "keras>=3.0.0",
            "tensorflow>=2.13.0",
        ],
        "torch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "all": [
            "keras>=3.0.0",
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
    },
)

# Made with Bob
