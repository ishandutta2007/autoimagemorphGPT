from setuptools import setup, find_packages

setup(
    name='autoimagemorphGPT',
    version='1.0.0',
    description='A sample Python package',
    author='Ishan Dutta',
    author_email='ishandutta2007@gmail.com',
    url='https://github.com/ishandutta2007/autoimagemorphGPT',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python>=4.5.1',
        # add other dependencies here
    ],
)


