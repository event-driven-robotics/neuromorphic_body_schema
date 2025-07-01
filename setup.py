from setuptools import setup, find_packages

setup(
    name="neuromorphic_body_schema",
    version="0.1.0",
    license='gpl',
    description="A package for neuromorphic body schema modeling of the iCub humanoid robot.",
    author="Simon F. Muller-Cleve",
    author_email="simon.mullercleve@iit.it",
    url='https://github.com/event-driven-robotics/neuromorphic_body_schema',
    keywords=['event', 'event camera', 'event-based', 'event-driven', 'spike', 'dvs', 'dynamic vision sensor',
              'bio inspired skin', 'neuromorphic skin', 'bio inspired proprioception', 
              'neuromorphic proprioception', 'neuromorphic', 'aer', 'address-event representation'],
    packages=find_packages(),
    install_requires=[
        "mujoco>=3.3.3",
        "opencv-python>=4.10.0.84",
        "numpy==2.0.0",
        "scipy==1.15.2"
    ],
    python_requires=">=3.10.12",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    include_package_data=True
)