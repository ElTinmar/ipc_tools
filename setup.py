from distutils.core import setup

setup(
    name='ipc_tools',
    author='Martin Privat',
    version='0.4.5',
    packages=['ipc_tools'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='share numpy arrays between processes',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "pyzmq",
        "arrayqueues",
        "opencv-python",
        "pandas",
        "seaborn",
        "tqdm",
        "matplotlib",
        "scipy",
        "multiprocessing_logger @ git+https://github.com/ElTinmar/multiprocessing_logger.git@main",
    ]
)