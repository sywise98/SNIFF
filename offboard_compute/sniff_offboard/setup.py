from setuptools import setup
import os
from glob import glob

from setuptools import setup, find_packages

package_name = 'sniff_offboard'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']), # This will find all packages including nodes
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='floridaman',
    maintainer_email='floridaman@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'item_extractor_node = sniff_offboard.nodes.item_extractor_node:main',
            'yolo_world_node = sniff_offboard.nodes.yolo_world_node:main',
        ],
    },
)
