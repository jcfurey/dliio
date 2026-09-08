#!/usr/bin/env python3
"""Retrieve, validate and optimize revisits into a new sealed mapping archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

import numpy as np

from dliio_mapping.core import Limits, Store, snapshot_database
from dliio_mapping.retrieval import propose, settings
import dliio_mapping


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--configuration', type=Path, required=True,
                        help='Explicit graph noise, failure policy and retrieval JSON')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--proposals-only', action='store_true', help='Save the candidate report and trial poses without an archive mutation')
    args = parser.parse_args()
    source, destination, report_path = (p.expanduser().resolve() for p in (args.source, args.destination, args.report))
    if source == destination or destination.exists() or report_path.exists():
        parser.error('Use new destination and report paths; source archives are never overwritten')
    if not destination.parent.is_dir() or not report_path.parent.is_dir():
        parser.error('Destination and report directories must already exist')
    config = settings(json.loads(args.configuration.read_text()))
    code_hashes = {path.name: digest(path) for path in Path(dliio_mapping.__file__).parent.glob('*.py')}
    code_hashes[Path(__file__).name] = digest(Path(__file__).resolve())
    import _dliio_pose_graph as native
    code_hashes[Path(native.__file__).name] = digest(Path(native.__file__).resolve())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    source_hash = digest(source)
    store = Store(source, Limits(resident_submaps=1))
    try:
        with report_path.with_suffix('.progress.jsonl').open('x') as progress_file:
            def progress(value):
                progress_file.write(json.dumps(value, allow_nan=False)+'\n')
                progress_file.flush()
                print(json.dumps({key: value[key] for key in ('from_id', 'to_id', 'seconds', 'status', 'reason') if key in value}), flush=True)
            report, poses = propose(store, config, progress)
        report.update(source=str(source), source_sha256=source_hash,
                      canonical_configuration_sha256=config_hash, code_sha256=code_hashes)
    finally:
        store.close()
    if digest(source) != source_hash:
        raise RuntimeError('Source archive changed during retrieval')
    report['source_unchanged'] = True
    np.savez_compressed(report_path.with_suffix('.poses.npz'), poses=poses)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if args.proposals_only or not report['loops']:
        print(json.dumps(dict(accepted_candidates=len(report['loops']), report=str(report_path))), flush=True)
        return
    with tempfile.TemporaryDirectory(prefix='.dliio-loops-', dir=destination.parent) as temporary:
        working = Path(temporary)/'working.dliomap'
        snapshot_database(source, working)
        store = Store(working, Limits(resident_submaps=1), editable=True)
        try:
            def request(action, **extra):
                return dict(session_id=store.meta['session_id'], expected_revision=store.meta['pose_revision'],
                            request_id=str(uuid.uuid4()), action=action, **extra)
            if not store.graph_stats()['graph_initialized']:
                result = store.update_graph(request('initialize', configuration=config['graph']))
                if result['status'] != 'accepted':
                    raise RuntimeError('Graph initialization rejected: '+json.dumps(result))
            result = store.update_graph(request('add_loops', loops=report['loops']))
            if result['status'] != 'accepted':
                raise RuntimeError('Loop batch rejected: '+json.dumps(result))
            store.save(destination)
            report.update(commit=result, destination=str(destination), destination_sha256=digest(destination))
        finally:
            store.close()
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(accepted_loops=len(report['loops']), destination=str(destination), report=str(report_path))), flush=True)


if __name__ == '__main__':
    main()
