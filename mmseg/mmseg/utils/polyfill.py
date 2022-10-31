import functools
import warnings

import mmcv
# import getpass


@functools.wraps(mmcv.symlink)
def _patched_mmcv_symlink(source, target, *, old_symlink=mmcv.symlink):
    # Azure's blobfuse does not allow symbol links, we then fallback
    # to noop
    try:
        return old_symlink(source, target)
    except OSError as exc:
        if exc.errno == 38:  # Function not implemented
            warnings.warn(f"Skip symlink {source} -> {target} due to OS unsupport")
        else:
            raise exc


mmcv.symlink = _patched_mmcv_symlink


@functools.wraps(mmcv.runner.utils.getuser)
def _patched_getuser(*, old_getuser=mmcv.runner.utils.getuser):
    try:
        return old_getuser()
    except KeyError:
        return "unknown"


mmcv.runner.utils.getuser = _patched_getuser
