from pathlib import Path
from setuptools import setup, find_packages


_info_ = {}
with open("heapy/__info__.py", "r") as f:
    exec(f.read(), _info_)
    
    
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8")


setup(
    name="heapyx",
    version=_info_['__version__'],
    description="High-energy (X-ray and gamma-ray) astronomical data analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
    packages=find_packages(exclude=["examples*"]),
    include_package_data=True,
    package_data={
        "heapy": [
            "docs/*",
            "docs/**/*",
        ]
    },
    project_urls={
        "Source Code": "https://github.com/jyangch/heapy"
    }
)
