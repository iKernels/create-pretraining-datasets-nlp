import setuptools
import json

def load_long_description():
    with open("README.md", "r") as fh:
        long_description = fh.read()
    return long_description

def load_requirements():
    requirements = []
    with open("requirements.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                requirements.append(line)
    return requirements



def get_version():
    from create_pretraining_dataset import __version__
    return __version__

setuptools.setup(
    name='create-pretraining-datasets-nlp',
    version=get_version(),
    description='Easily create large NLP datasets to pre-train language models',
    long_description=load_long_description(),
    url='git@github.com:ikernels/create-pretraining-datasets-nlp.git',
    author='Luca Di Liello',
    author_email='luca.diliello@unitn.it',
    license='GNU v2',
    packages=setuptools.find_packages(),
    install_requires=load_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU v2 License",
        "Operating System :: OS Independent",
    ]
)