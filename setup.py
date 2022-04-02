import setuptools
import os

# gather dependencies from requirements.txt
lib_folder = os.path.dirname(os.path.realpath(__file__))
reqquirements_filename = lib_folder + "/requirements.txt"
install_requires = []
if os.path.isfile(reqquirements_filename):
    with open(reqquirements_filename) as f:
        install_requires = f.read().splitlines()
install_requires_n = []
for x in install_requires:
    if x != "" and x[0] != "#":
        install_requires_n.append(x)
install_requires = install_requires_n
# print(install_requires)

setuptools.setup(
    name="goat",  # Replace with your username
    version="1.0.0",
    author="ethanweber",
    author_email="ethanweber@berkeley.edu",
    description="A short description.",
    long_description="A long description.",
    long_description_content_type="text/markdown",
    url="https://github.com/ethanweber/goat",
    packages=setuptools.find_packages(),
    test_suite="goat",
    python_requires=">=3.6",
    install_requires=install_requires,
)
