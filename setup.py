from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xamp-predictor",
    version="1.0.0",
    author="XAMP Development Team",
    author_email="jing.li@sjtu.edu.cn",
    description="A dual-engine deep learning framework for antimicrobial peptide prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Li-Lab-SJTU/XAMP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pandas",
        "numpy",
        "scikit-learn",
        "fair-esm",
    ],
    entry_points={
        "console_scripts": [
            "xamp-predict=predict_command:main",
        ],
    },
)