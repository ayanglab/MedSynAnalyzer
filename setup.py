from setuptools import setup, find_packages

setup(
    name='synthetic_image_analyzer',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        # Add any other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'synthetic_image_analyzer = synthetic_image_analyzer.main:main',
        ]},
    author='Xiaodan Xing',
    description='A comprehensive analysis tool that offers a wide range of functionalities for synthetic image quality analysis. '
                'It includes features for feature extraction, quality evaluation, fidelity computation, '
                'and more. With its comprehensive set of tools, this package empowers users to perform in-depth analysis and assessment '
                'of synthetic images, enabling better understanding and evaluation of their quality.',
)