"""Связь управляющего манифеста DoE с месячным откликом."""

from datetime import date

from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from scripts.build_dataset import control_roles


def test_reference_schedule_preserves_role_when_scenario_response_is_zero():
    steps = [
        ScheduleStep(
            date(2014, 1, 1),
            (ProducerControl("9", "OPEN", "LRAT", 30.0, 50.0),),
            (InjectorControl("9", "WATER", "SHUT", "RATE", None, 300.0),),
        ),
        ScheduleStep(
            date(2014, 2, 1),
            (ProducerControl("9", "SHUT", "LRAT", None, 50.0),),
            (InjectorControl("9", "WATER", "OPEN", "RATE", 40.0, 300.0),),
        ),
    ]

    roles = control_roles(steps, {"2014-01-01", "2014-02-01", "2014-03-01"})

    assert roles[("2014-01-01", "9")] == "production"
    assert roles[("2014-02-01", "9")] == "injection"
    assert roles[("2014-03-01", "9")] == "injection"
