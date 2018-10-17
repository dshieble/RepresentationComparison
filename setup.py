from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
   name='repcomp',
   version='0.1',
   author="Dan Shiebler",
   author_email="danshiebler@gmail.com",
   description="A package for comparing trained embedding models.",
   long_description=long_description,
   long_description_content_type="text/markdown",
   url="https://github.com/pypa/sampleproject",
   packages=['repcomp'],
   classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent"
    ],
   install_requires=[
    'numpy',
    'pandas',
    'tensorflow',
    'gensim',
    'keras',
    'annoy',
    'mock',
    'scipy',
    'tqdm',
    'scikit-learn',
    'pytest'])
