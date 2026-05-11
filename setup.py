from distutils.core import setup

setup(
    name='ipc_tools',
    python_requires='>=3.8',
    author='Martin Privat',
    version='0.4.7',
    packages=['ipc_tools'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='share numpy arrays between processes',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "pyzmq",
        "arrayqueues",
        "pandas",
        "seaborn",
        "tqdm",
        "matplotlib",
        "scipy",
        "multiprocessing_logger @ git+https://github.com/ElTinmar/multiprocessing_logger.git@v0.3.12",
    ]
)