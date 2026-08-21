"""Пересчёт справки Manus по ППД (`ppd-capacity.md`) на фактические числа Модели Z.

В промпте задачи №3 (`docs/manus-tasks-round2.md`) закачка 11 671 / 17 254 тыс. м³
была названа годовой, хотя это накопленные объёмы за расчётное окно с 2007 года.
Отсюда в справке 32 → 47 тыс. м³/сут и 100 нагнетательных скважин. Здесь те же
формулы (`ppd-capacity-calc.py`, физика не менялась) считаются по выгрузке
базового прогона.

Запуск: python docs/manus/ppd-rescale.py
"""

from math import pi

import pandas as pd

RHO = 1050.0        # кг/м³, минерализованная вода — допущение справки Manus
G = 9.81
ETA_PUMP = 0.76     # паспортный минимум ЦНСА-180
ETA_MOTOR = 0.95
HOURS_YEAR = 8760

WINDOW_START = "2007-01-01"
SCENARIO = 17253.7 / 11671.0   # `runs/inject20`, закачка ×2.0 к уставкам базы

rows = pd.read_csv("runs/base/rows.csv", parse_dates=["DATA"])
win = rows[rows.DATA >= WINDOW_START]
inj = win[win.WWIR > 0]

months = win.DATA.nunique()
cum_m3 = win.WWIT_Diff.sum()
last_year = int(inj.DATA.dt.year.max()) - 1          # последний полный год окна
q_last = inj[inj.DATA.dt.year == last_year].WWIR.sum() / 12
q_mean = cum_m3 / (months * 30.44)
n_wells = inj.well.nunique()

print(f"окно {win.DATA.min():%Y-%m} … {win.DATA.max():%Y-%m}, {months} мес")
print(f"накопленная закачка: {cum_m3/1000:,.1f} тыс. м³ за окно")
print(f"средняя по окну: {q_mean:,.0f} м³/сут; за {last_year} год: {q_last:,.0f} м³/сут")
print(f"нагнетательных скважин: {n_wells}; приёмистость среднее {inj.WWIR.mean():.0f}, "
      f"p90 {inj.WWIR.quantile(0.9):.0f}, макс {inj.WWIR.max():.0f} м³/сут")
print(f"устьевое давление нагнетания: медиана {inj.THP.median():.0f}, "
      f"p90 {inj.THP.quantile(0.9):.0f}, макс {inj.THP.max():.0f} атм")
print()

q0 = q_last
q1 = q0 * SCENARIO
print(f"сценарий ×{SCENARIO:.2f}: {q0:,.0f} → {q1:,.0f} м³/сут (+{q1-q0:,.0f}); "
      f"годовой прирост {(q1-q0)*365/1e6:.2f} млн м³/год")
print(f"на скважину при неизменном фонде: {q0/n_wells:.0f} → {q1/n_wells:.0f} м³/сут "
      f"(исторический максимум одной скважины {inj.WWIR.max():.0f})")
print()

# Напор берём по фактическому устьевому давлению, а не по иллюстративным 1200 м справки.
for thp_atm in [inj.THP.median(), inj.THP.quantile(0.9), 300.0]:
    head = thp_atm * 101325 / (RHO * G)
    print(f"устье {thp_atm:.0f} атм → напор {head:,.0f} м ({thp_atm*0.101325:.1f} МПа)")
    for label, q in [("текущая", q0), ("сценарий", q1), ("прирост", q1 - q0)]:
        p_hyd = RHO * G * (q / 86400) * head / 1000
        p_el = p_hyd / (ETA_PUMP * ETA_MOTOR)
        print(f"  {label}: {q/24:,.0f} м³/ч; N_гидр={p_hyd:,.0f} кВт; "
              f"N_эл={p_el:,.0f} кВт; E={p_el*HOURS_YEAR/1e6:.2f} ГВт·ч/год")
print()

for q_pump_h in [63.0, 90.0, 180.0]:
    print(f"ЦНС {q_pump_h:.0f} м³/ч: текущая {q0/24/q_pump_h:.2f} агрегата, "
          f"сценарий {q1/24/q_pump_h:.2f}, прирост {(q1-q0)/24/q_pump_h:.2f}")
print()

for v in [1.0, 1.5]:
    d = (4 * (q1 / 86400) / (pi * v))**0.5
    print(f"минимальный внутренний диаметр коллектора при {q1:,.0f} м³/сут и v≤{v} м/с: {d*1000:.0f} мм")
