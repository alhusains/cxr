from setuptools import setup, find_packages

setup(
    name='cxr_classification',
    version='0.1.0',
    description='Chest X-Ray Classification System',
    author='Machine Learning Specialist Candidate',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        line.strip() 
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    entry_points={
        'console_scripts': [
            'cxr-train=src.models.train:main',
            'cxr-evaluate=src.evaluation.evaluate:main',
            'cxr-serve=src.deployment.api:main',
        ],
    },
)
