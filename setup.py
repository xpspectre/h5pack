import setuptools
import h5pack

main_package = 'h5pack'


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name=main_package,
    version=h5pack.__version__,
    author='Kevin Shi',
    author_email='xpspectre@gmail.com',
    description='HDF5 Easy Serialization Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/xpspectre/h5pack',
    platforms=['any'],
    license='MIT License',
    packages=[main_package],
    install_requires=[
        'h5py>=2.9'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Environment :: Win32 (MS Windows)',
        'Environment :: X11 Applications',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)
