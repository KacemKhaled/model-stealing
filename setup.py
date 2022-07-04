import setuptools

setuptools.setup(
    name="ms",
    version="0.1",
    author="Kacem Khaled",
    url="https://github.com/KacemKhaled/model-stealing",
    packages=setuptools.find_packages(include=['ms', 'ms.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
)