from setuptools import setup, find_packages


_info_ = {}
with open("heapy/__info__.py", "r") as f:
    exec(f.read(), _info_)


setup(
    name="heapy",
    version=_info_['__version__'],
    description="Astronomical data analysis tool",
    long_description="A unified toolkit for timing and spectral analysis of X-ray and gamma-ray transient data",
    author="Jun Yang",
    author_email="jyang@smail.nju.edu.cn",
    url="https://github.com/jyangch/heapy",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.4",
        "pandas>=1.4.2",
        "scipy>=1.8.0",
        "astropy>=5.2.2",
        "matplotlib>=3.2.1",
        "tqdm>=4.64.1",
        "plotly>=5.22.0",
        "pybaselines>=1.1.0"
    ],
    packages=find_packages(exclude=["examples*", "docs*"]),
    include_package_data=True,
    project_urls={
        "Source Code": "https://github.com/jyangch/heapy"
    }
)
