
# run 'pip install -e .' in the command line for installation

from setuptools import setup

setup(name="simple-mpi",
      version="0.1",
      description="a simple master slave style wrapper for mpi",
      url="not yet there",
      author="Jobst Heitzig, Jonathan Donges, pyunicorn authors, and Tim Kittel",
      license="BSD (3-clause)",
      packages=["simple_mpi"],
      install_requires=[
          "mpi4py>=2.0.0",
      ],
      zip_safe=False)


