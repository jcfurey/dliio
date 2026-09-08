"""Read-only completion check for a stopped sensor stream and its loop worker."""
def drained(mapping, loops=None):
    if (not mapping or int(mapping.get('queue_depth', -1)) != 0 or
            mapping.get('pose_revision') != mapping.get('cached_pose_revision')):
        return False
    if loops is None:
        return True
    return (loops.get('loop_idle') == 'True' and not loops.get('loop_error') and
            loops.get('session_id') == mapping.get('session_id') and
            int(loops.get('loop_checked_observations', -1)) >= int(mapping['keyframes']) and
            loops.get('loop_checked_pose_revision') == mapping.get('pose_revision'))
