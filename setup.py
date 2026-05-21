from setuptools import setup, find_packages

setup(
    name="medrl",
    version="0.1.0",
    description="Medical Long-Form Reasoning Alignment via GRPO",
    author="MedRL Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "deepspeed>=0.12.0",
        "accelerate>=0.25.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "pre-commit>=3.0.0",
        ],
    },
)
