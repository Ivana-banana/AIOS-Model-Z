"""План Шепли не должен считать коалиции, которые ничего не меняют."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from aios.groups import Group

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "shapley_plan.py"
SPEC = importlib.util.spec_from_file_location("shapley_plan", SCRIPT)
assert SPEC and SPEC.loader
shapley_plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shapley_plan)


def test_active_groups_excludes_groups_without_controlled_wells():
    groups = [
        Group("добывающие сухие", "producer", {"1": 1.0}),
        Group("добывающие обводнённые", "producer", {}),
        Group("закачка в невыработанное", "injector", {"2": 1.0}),
        Group("закачка в обводнённое", "injector", {}),
    ]

    active = shapley_plan.active_groups(groups)

    assert [group.name for group in active] == [
        "добывающие сухие", "закачка в невыработанное",
    ]
