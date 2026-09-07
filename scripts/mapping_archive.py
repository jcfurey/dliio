#!/usr/bin/env python3
"""Inspect, checkpoint, or export dliio mapping archives without running ROS."""
import argparse
import json
from pathlib import Path
import tempfile
from dliio_mapping.core import Limits, Store, snapshot_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['inspect', 'snapshot', 'export', 'revise', 'restore'])
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path, nargs='?')
    parser.add_argument('--max-voxels', type=int, default=Limits().max_voxels,
                        help='Per-submap point read limit (also applies when voxel filtering is disabled)')
    parser.add_argument('--max-input-points', type=int, default=Limits().max_input_points, help='Per-keyframe read limit')
    parser.add_argument('--fusion-size', type=float, default=0.,
                        help='Global export fusion spacing in metres; 0 exports the stored samples')
    parser.add_argument('--revision', type=Path, help='JSON revision request; required for revise/restore')
    args = parser.parse_args()
    if args.operation in ('snapshot', 'export', 'revise', 'restore') and args.destination is None:
        parser.error('This operation requires an absolute destination filename')
    if args.operation == 'snapshot':
        snapshot_database(args.source, args.destination)
        return
    if args.operation in ('revise', 'restore'):
        if args.revision is None:
            parser.error('revise/restore requires --revision request.json')
        if not args.destination.is_absolute() or not args.destination.parent.is_dir() or args.destination.exists():
            parser.error('Destination must be a new absolute filename in an existing directory')
        if args.revision.stat().st_size > 64 * 1024 * 1024:
            parser.error('Revision JSON exceeds 64 MiB')
        request = json.loads(args.revision.read_text())
        with tempfile.TemporaryDirectory(prefix='.dliio-revision-', dir=args.destination.parent) as temporary:
            working = Path(temporary) / 'working.dliomap'
            snapshot_database(args.source, working)
            store = Store(working, Limits(max_voxels=args.max_voxels, max_input_points=args.max_input_points), editable=True)
            try:
                if args.operation == 'revise':
                    if request.get('frame_id') != store.meta['map_frame']:
                        raise ValueError('Revision frame_id differs from archive map frame')
                    store.apply_revision(request['session_id'], request['expected_revision'],
                        [(entry['id'], entry['pose']) for entry in request['poses']],
                        request_id=request['request_id'], reason=request['reason'])
                else:
                    store.restore_revision(request['session_id'], request['expected_revision'], request['target_revision'],
                                           request_id=request['request_id'])
                store.save(args.destination)
                print(json.dumps(dict(destination=str(args.destination), pose_revision=store.meta['pose_revision'])))
            finally:
                store.close()
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
