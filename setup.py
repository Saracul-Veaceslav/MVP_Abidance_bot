from setuptools import setup, find_packages

setup(
    name="abidance",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "pandas>=2.0.2",
        "numpy>=1.24.3",
        "python-dotenv>=1.0.0",
        "ccxt>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "abidance=abidance.cli.main:main",
        ],
    },
    author="Abidance Team",
    author_email="example@example.com",
    description="A cryptocurrency trading bot",
    keywords="crypto, trading, bot, finance",
    python_requires=">=3.8",
) 