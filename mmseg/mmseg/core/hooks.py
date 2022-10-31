from mmcv.runner.hooks.logger.base import LoggerHook
from mmcv.runner.hooks.hook import HOOKS

from contextlib import suppress

try:
    from azureml.core import Run

    aml_ctx = Run.get_context()
except ImportError:

    def noop(*args, **kwds):
        ...

    class aml_ctx:
        log = staticmethod(noop)


@HOOKS.register_module()
class AMLLoggerHook(LoggerHook):
    def log(self, runner):
        with suppress(Exception):
            output = runner.log_buffer.output
            mode = self.get_mode(runner)
            prefix = mode + "/"
            if runner.rank == 0:
                keys = list(output)[:10]
                for key in keys:
                    value = output[key]
                    aml_ctx.log(name=prefix + key, value=value)

        return output
