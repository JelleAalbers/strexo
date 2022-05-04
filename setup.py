import setuptools

with open("requirements.txt") as f:
    requires = [x.strip() for x in f.readlines()]

with open("README.md") as file:
    readme = file.read()

with open("HISTORY.md") as file:
    history = file.read()

setuptools.setup(
    name="strexo",
    version="0.0.0",
    description="Toy strax prototype for nEXO",
    url="https://github.com/JelleAalbers/strexo",
    python_requires=">=3.6",
    setup_requires=["pytest-runner"],
    install_requires=requires,
    tests_require=["pytest"],
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme", "matplotlib", "numpydoc", "recommonmark"]
    },
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    zip_safe=False,
)
