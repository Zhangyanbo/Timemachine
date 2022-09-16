import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="time_machine",
    version="0.0.1",
    author="Yanbo Zhang",
    author_email="Zhang.Yanbo@asu.edu",
    description="A package for time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zhangyanbo/Timemachine",
    project_urls={
        "Bug Tracker": "https://github.com/Zhangyanbo/Timemachine/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['time_machine'],
    python_requires=">=3.6",
    test_suite='test',
)