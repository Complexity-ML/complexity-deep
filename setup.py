"""
Complexity Deep - Llama + Full INL Dynamics (Robotics)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="complexity-deep",
    version="0.7.1",
    description="Multicouche robotics architecture with KQV + INL Dynamics + Token-Routed MLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pacific Prime",
    author_email="contact@pacific-prime.ai",
    url="https://github.com/Web3-League/complexity-deep",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "tokenizers>=0.13.0",
        "safetensors>=0.3.0",
        "tqdm",
        "tensorboard",
    ],
    extras_require={
        "cuda": ["triton>=2.0.0"],
        "dev": ["pytest", "black", "isort"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm transformer dynamics robotics llama pytorch deep-learning velocity-control",
)
