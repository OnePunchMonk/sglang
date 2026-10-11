"""launch_server hands scheduler args to pre-started workers when the caller registered them."""

import sglang.multimodal_gen.runtime.launch_server as launch_mod


class _FakeProcess:
    def __init__(self, args, name):
        self.args = args
        self.name = name


def test_registered_workers_are_used_in_order(monkeypatch):
    monkeypatch.setattr(launch_mod, "_prespawned_scheduler_factories", [])
    made = []

    def factory(args, name):
        made.append((args, name))
        return _FakeProcess(args, name)

    launch_mod.register_prespawned_schedulers([factory, factory])
    first = launch_mod._new_scheduler_process(args=(0, 0), name="w0")
    second = launch_mod._new_scheduler_process(args=(1, 1), name="w1")

    assert [m[1] for m in made] == ["w0", "w1"]
    assert first.args == (0, 0) and second.name == "w1"
    assert launch_mod._prespawned_scheduler_factories == []


def test_falls_back_to_a_new_process_without_registered_workers(monkeypatch):
    monkeypatch.setattr(launch_mod, "_prespawned_scheduler_factories", [])
    process = launch_mod._new_scheduler_process(args=(0, 0), name="w0")
    assert process.name == "w0"
    assert process._target is launch_mod.run_scheduler_process
    assert process.daemon
