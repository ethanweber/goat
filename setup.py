import setuptools

setuptools.setup(
    name="goat",  # Replace with your username
    version="1.0.0",
    author="ethanweber",
    author_email="ethanweber@berkeley.edu",
    description="A short description.",
    long_description="A long description.",
    long_description_content_type="text/markdown",
    url="https://github.com/ethanweber/goat",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    test_suite="goat",
    python_requires='>=3.6',
)
