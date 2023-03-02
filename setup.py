import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlc",
    version="0.0.1",
    author="qiaoliang",
    author_email="ql1an9@mail.ustc.edu.cn",
    description="PyTorch -> tvm Relax --(op fuse & op optimizer)--> tvm TensorIR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiaolian9/mlc",
    install_requires=["loguru"],
    project_urls={
        "Bug Tracker": "https://github.com/qiaolian9/mlc/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: MLC & DL",
    ],
    packages=setuptools.find_packages(exclude=["doc", "test", "examples"]),
    python_requires=">=3.6",
    keywords="Pytorch, MLC",
)