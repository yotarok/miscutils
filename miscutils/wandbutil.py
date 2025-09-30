class NullWandbRun:
    def log(self, *unused_args, **unused_kwargs):
        pass

    def watch(self, *unused_args, **unused_kwargs):
        pass

    def finish(self, *unused_args, **unused_kwargs):
        pass

    def save(self, *unused_args, **unused_kwargs):
        pass

    @property
    def config(self):
        return {}
