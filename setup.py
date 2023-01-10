from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'sympy',
]

setup(
    name='kdl_sympy',
    version='0.0.0',
    author='dohi',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=install_requires,
)
