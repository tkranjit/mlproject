from typing import List
from setuptools import setup, find_packages

def get_requirements(file_path: str) -> List[str]:
    '''
    this function return the list of requirements

    '''

    HYPHON_E_DOT = '-e .'
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPHON_E_DOT in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='mlproject',
    version='0.1',  
    description='A simple machine learning project',
    author='ranjith', 
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)