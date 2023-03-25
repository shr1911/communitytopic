import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='communitytopic',
    version='0.1',
    author='Shraddha Makwana',
    description='Community Topic - Topic Modelling Method',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    url='https://github.com/shr1911/communitytopic',
    author_email='smakwana@ualberta.ca',
    install_requires=[
        'setuptools~=67.6.0',
        'spacy~=3.5.0',
        'numpy~=1.21.5',
        'gensim~=4.2.0',
        'networkx~=2.8.4',
        'igraph~=0.10.4'
    ]
)
