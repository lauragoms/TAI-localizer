from setuptools import find_packages, setup

setup(
    name="tai_localiser",
    version="0.0",
    description="Peru and Lauras fun code",
    long_description="",
    author="Laura and Peru",
    author_email="peru.dornellas@gmail.com",
    license="Apache Software License",
    home_page="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2",
        "scipy",
        "matplotlib",
        "flake8",
        "pytest",
        "pytest-cov",
        "pytest-xdist",
        "nbmake",
        "pytest-github-actions-annotate-failures",
        "mpire",
        "jax",
    ],
)
