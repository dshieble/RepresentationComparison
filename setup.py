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
    'pandas>=0.20.3',
    'gensim>=0.13.3',
    'keras>=2.0.1',
    'annoy>=1.12.0',
    'mock>=2.0.0',
    'tqdm>=4.19.5',
    'scikit-learn>=0.19.1',
    'pytest>=3.4.1'])
