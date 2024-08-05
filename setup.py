# this module help us to easy installation of packages
from setuptools import find_packages, setup
from typing import List

# there is  -e . in requirements which help us to automatic trigger the setup.py
HYPEN_e_rem = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    this function will return list of all requirements
    """
    requirement = []
    fobj = open("requirements.txt", 'r')
    requires = fobj.readlines()
    requirements = [req.replace("\n", "") for req in requires]

    # write the codition for removing the -e .
    if HYPEN_e_rem in requirements:
        requirements.remove(HYPEN_e_rem)

        # returning the removed -e .
    return requirements


# setup
setup(
    name="MLproject",
    version="0.0.1",
    author="Rohit",
    author_email="katkarrohit203@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)