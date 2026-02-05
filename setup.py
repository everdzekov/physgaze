from setuptools import setup, find_packages

setup(
    name="physgaze",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'tqdm>=4.62.0',
        'scipy>=1.7.0',
        'h5py>=3.3.0',
        'scikit-learn>=0.24.0',
        'opencv-python>=4.5.0',
        'tensorboard>=2.7.0',
        'pyyaml>=5.4.0',
    ],
    python_requires='>=3.7',
)