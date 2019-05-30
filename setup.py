import os
from setuptools import setup, Extension, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name='WFC3_Back_Sub',
	version='1.0',
	description='WFC3 IR Grism Background Subtraction Module',
	author='Nor Pirzkal',
	author_email='npirzkal@stsci.edu',
	package_dir = {
        'WFC3_Back_Sub': 'WFC3_Back_Sub'},
    packages=['WFC3_Back_Sub',],
    package_data={'WFC3_Back_Sub': ['data/*.fits']},
)