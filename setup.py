from setuptools import setup, find_packages

setup(
    name="hsuga",
    version="1.0.0",
    description="HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.14.0",
        "pyyaml>=6.0",
    ],
)
