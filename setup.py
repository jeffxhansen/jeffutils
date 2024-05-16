from setuptools import setup, find_packages
setup(
    name='jeffutils',
    version='0.5.7',
    author='Jeff Hansen',
    author_email='jeffxhansen@gmail.com',
    description='A series of useful functions and decorators I use in most of my projects. Feel free to use them as well :)',
    package_dir={"": "src"},
    packages = find_packages(where="src"),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
