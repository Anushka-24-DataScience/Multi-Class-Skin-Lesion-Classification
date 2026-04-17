from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as f:
        requirements = [req.strip() for req in f.readlines()]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="DermaCancerScan",
    version="0.0.1",
    author="Anushka Srivastava",
    author_email="anuskasrivastav24@gmail.com",
    package_dir={"": "src"},           
    packages=find_packages(where="src"), 
    install_requires=get_requirements("requirements.txt"),
)