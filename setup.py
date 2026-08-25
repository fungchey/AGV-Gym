import setuptools

setuptools.setup(
    name="agv-gym",
    version="1.0.0",
    description="AGV-Gym: Deep reinforcement learning environment for battery-constrained automated guided vehicles scheduling problems",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "gymnasium>=0.28.0",
        "matplotlib",
        "numpy",
        "pandas",
        "Pillow",
        "scipy"
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    include_package_data=True,
    package_data={'pydispatching': ['agv_data/*']},
    python_requires=">=3.7",
)
