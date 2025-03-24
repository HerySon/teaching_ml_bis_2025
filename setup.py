from setuptools import find_packages, setup

setup(
    name='clustering_OFF',
    version='1.0',
    description='Teaching project for data science & AI classes',
    author='Your Name',
    packages=['scripts', 'scripts.data', 'scripts.features', 'scripts.models', 'scripts.visualization'],
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'jupyter>=1.0.0',
        'ipykernel>=6.0.0'
    ],
    include_package_data=True,
    zip_safe=False
)
