#!/usr/bin/env python3
"""Build dliio_offline_replay against an existing tested CMake build tree."""
import argparse
from pathlib import Path
import shlex
import subprocess

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('build', type=Path)
ap.add_argument('output', type=Path)
ap.add_argument('--source', type=Path, default=Path(__file__).with_name('dliio_offline_replay.cpp'),
                help='Alternative standalone C++ harness linked against the same node build')
a = ap.parse_args()
build, output = a.build.resolve(), a.output.resolve()
target = build / 'CMakeFiles/test_pointcloud_channels.dir'
flags = []
for line in (target / 'flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
        flags.extend(shlex.split(line.split('=', 1)[1]))
source = a.source.resolve()
obj = str(output) + '.o'
subprocess.run(['c++', *flags, '-c', str(source), '-o', obj], cwd=build, check=True)
link = shlex.split((target / 'link.txt').read_text())
link = [obj if s.endswith('/test/test_pointcloud_channels.cpp.o') else s for s in link]
link[link.index('-o') + 1] = str(output)
subprocess.run(link, cwd=build, check=True)
print(output)
