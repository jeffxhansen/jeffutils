from setuptools import setup, find_packages
setup(
    name='jeffutils',
    version='0.8.8',
    author='Jeff Hansen',
    author_email='jeffxhansen@gmail.com',
    description="Welcome to Jeff Hansen's suite of useful python functions! I use lots of these functions on most of my Data Analysis, Backend-Dev, and Machine Learning projects, and I hope you find some of them useful as well!",
    package_dir={"": "src"},
    packages = find_packages(where="src"),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    url="https://github.com/jeffxhansen/jeffutils",
    python_requires='>=3.6',
)
