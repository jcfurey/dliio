#!/usr/bin/env python3
"""Inspect, checkpoint, or export dliio mapping archives without running ROS."""
import argparse
import json
from pathlib import Path
from dliio_mapping.core import Limits, Store, snapshot_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['inspect', 'snapshot', 'export'])
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path, nargs='?')
    parser.add_argument('--max-voxels', type=int, default=Limits().max_voxels,
                        help='Per-submap point read limit (also applies when voxel filtering is disabled)')
    parser.add_argument('--max-input-points', type=int, default=Limits().max_input_points, help='Per-keyframe read limit')
    parser.add_argument('--fusion-size', type=float, default=0.,
                        help='Global export fusion spacing in metres; 0 exports the stored samples')
    args = parser.parse_args()
    if args.operation in ('snapshot', 'export') and args.destination is None:
        parser.error('This operation requires an absolute destination filename')
    if args.operation == 'snapshot':
        snapshot_database(args.source, args.destination)
        return
    store = Store(args.source, Limits(max_voxels=args.max_voxels, max_input_points=args.max_input_points))
    try:
        if args.operation == 'inspect':
            print(json.dumps(dict(metadata=store.meta, **store.stats()), indent=2, allow_nan=False))
        else:
            print(json.dumps({'exported_points': store.export_pcd(args.destination, args.fusion_size or None)}))
    finally:
        store.close()


if __name__ == '__main__':
    main()
