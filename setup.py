import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Torch2Tensor",
    version="0.0.1",
    author="qiaoliang",
    author_email="ql1an9@mail.ustc.edu.cn",
    description="A easy tool for generating Tensor Program from Torch(besd on Torch FX & TVM Relax)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiaolian9/Torch2Tensor",
    install_requires=[
        "loguru",
        'tabulate',
        'torch',
        'torchvision',
        'xgboost==1.5',
        'mlc-ai-nightly-cu113'
        ],
    dependency_links=[
        'https://pypi.python.org/simple',
        'https://mlc.ai/wheels',
    ],
    project_urls={
        "Bug Tracker": "https://github.com/qiaolian9/Torch2Tensor/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: MLC & DL",
    ],
    packages=setuptools.find_packages(exclude=["doc", "test", "examples"]),
    python_requires=">=3.6",
    keywords="Pytorch, MLC, TVM",
)