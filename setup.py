import setuptools

# Read the contents of README.md file for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Specify dependencies
requirements = [
    "scikit-learn",
    "keras",
    "tensorflow",
    "tensorflow-privacy",
    "keras-tuner"
]

# Define package metadata
setuptools.setup(
    name="causal_inference_package",
    version="0.1",
    author="Your Name",
    author_email="your_email@example.com",
    description="A package for causal inference methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/causal_inference_package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
