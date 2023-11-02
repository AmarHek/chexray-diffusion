from setuptools import setup, find_packages

setup(
    name='cheff',
    version='0.0.1',
    description='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)