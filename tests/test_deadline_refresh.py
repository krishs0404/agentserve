from agentserve.engine.policies import TrajectoryDeadlinePolicy
from agentserve.engine.request import Request
from agentserve.engine.scheduler import Scheduler


def test_waiting_request_gains_deadline_pressure_without_new_arrival(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("agentserve.engine.policies.monotonic", lambda: clock[0])
    scheduler = Scheduler(
        policy=TrajectoryDeadlinePolicy(),
        max_prefill_per_step=1,
        enable_preemption=False,
        enable_overflow=False,
    )
    near_complete = Request(
        prompt="a",
        token_ids=[1],
        total_steps=2,
        step_index=1,
        estimated_output_tokens=20,
        deadline=1000,
    )
    urgent_later = Request(
        prompt="b",
        token_ids=[1],
        total_steps=2,
        step_index=0,
        estimated_output_tokens=100,
        deadline=10,
    )
    scheduler.add(near_complete)
    scheduler.add(urgent_later)
    assert scheduler.pending[0] is near_complete
    clock[0] = 5.0
    assert scheduler.get_prefill_batch() == [urgent_later]
