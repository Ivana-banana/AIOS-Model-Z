from math import pi

rho = 1050.0       # kg/m3 — минерализованная пластовая вода, инженерное допущение
rho_water = 1000.0 # kg/m3 — для паспортного сопоставления
g = 9.81           # m/s2
eta_pump = 0.76    # паспортный минимум для ЦНСА-180
eta_motor = 0.95   # инженерное допущение для крупного электродвигателя
availability = 0.95 # эксплуатационный коэффициент использования во времени
hours_year = 8760

q0_day = 32000.0
q1_day = 47000.0
dq_day = q1_day - q0_day

# Reservoir / injection wells
n_wells = 100
for label, q in [("Исходная закачка", q0_day), ("После увеличения", q1_day), ("Прирост", dq_day)]:
    print(f"{label}: {q:,.0f} m3/day; {q / 24:,.2f} m3/h; {q / 86400:,.6f} m3/s; {q / n_wells:,.0f} m3/day/well")
print(f"Annual added injection (365 d): {dq_day * 365:,.0f} m3/year")
print()

# Pump train selection, CНСА-180-1200m duty case
q_pump_h = 180.0
q_pump_day = q_pump_h * 24
q_pump_s = q_pump_h / 3600
head_case_m = 1200.0

hydraulic_per_pump_kw = rho * g * q_pump_s * head_case_m / 1000
electric_per_pump_kw = hydraulic_per_pump_kw / (eta_pump * eta_motor)
print(f"CNS-180 pump duty: Q={q_pump_h:.0f} m3/h, H={head_case_m:.0f} m")
print(f"Hydraulic power per running pump: {hydraulic_per_pump_kw:.1f} kW")
print(f"Electric input per running pump (eta_p={eta_pump:.2f}, eta_m={eta_motor:.2f}): {electric_per_pump_kw:.1f} kW")
print()

for label, q in [("Initial", q0_day), ("Final", q1_day), ("Increment", dq_day)]:
    n_equiv = q / q_pump_day
    n_running = int(-(-q // q_pump_day)) # ceiling for positive q
    flow_capacity = n_running * q_pump_day
    hyd_power = rho * g * (q / 86400) * head_case_m / 1000
    elec_power = hyd_power / (eta_pump * eta_motor)
    energy_24_7 = elec_power * hours_year / 1e6
    flow_when_available = q / availability
    electric_when_available = elec_power / availability
    energy_at_avail_same_volume = electric_when_available * (hours_year * availability) / 1e6
    print(f"{label}: pump-equivalents={n_equiv:.2f}; operating units required={n_running}; installed process capacity={flow_capacity:,.0f} m3/d")
    print(f"  Hydraulic power={hyd_power/1000:.3f} MW; electric input={elec_power/1000:.3f} MW")
    print(f"  Energy for stated annual volume={energy_24_7:.2f} GWh/y")
    print(f"  If availability is {availability:.0%} and the same annual volume must be maintained: operating flow={flow_when_available:,.0f} m3/d; P_el while running={electric_when_available/1000:.3f} MW; energy={energy_at_avail_same_volume:.2f} GWh/y")
print()

# Sensitivity by total required differential head at final and incremental flow
for head_m in [1000.0, 1200.0, 1500.0]:
    dp_mpa = rho * g * head_m / 1e6
    print(f"Head {head_m:.0f} m -> pressure equivalent {dp_mpa:.2f} MPa")
    for label, q in [("Final", q1_day), ("Increment", dq_day)]:
        phyd_kw = rho * g * (q / 86400) * head_m / 1000
        pel_kw = phyd_kw / (eta_pump * eta_motor)
        energy_gwh = pel_kw * hours_year / 1e6
        print(f"  {label}: P_hyd={phyd_kw/1000:.3f} MW; P_el={pel_kw/1000:.3f} MW; E_24/7={energy_gwh:.2f} GWh/y")
print()

# Pipeline internal diameters, velocity capacity and friction (Darcy-Weisbach)
q0_s = q0_day / 86400
q1_s = q1_day / 86400
lam = 0.025  # indicative Darcy factor (not a project value)
length_m = 10000.0
for d in [0.6, 0.7, 0.8, 0.9, 1.0]:
    area = pi * d**2 / 4
    v0 = q0_s / area
    v1 = q1_s / area
    # friction at final flow, without local losses and elevation
    dp_pa = lam * (length_m / d) * (rho * v1**2 / 2)
    print(f"ID {d*1000:.0f} mm: v at 32k={v0:.3f} m/s; v at 47k={v1:.3f} m/s; friction at 47k={dp_pa/1e6:.3f} MPa per 10 km")
print()

for v_design in [1.0, 1.5]:
    d_req = (4 * q1_s / (pi * v_design))**0.5
    print(f"Minimum internal diameter at final flow Q=47,000 m3/d for v<={v_design:.1f} m/s: {d_req*1000:.0f} mm")

# Fitting water rate against two project liquid capacity examples
for liquid_capacity in [20000.0, 23000.0]:
    overload = q1_day/liquid_capacity
    print(f"If incoming liquid increases by 50% to 47,000 m3/d against capacity {liquid_capacity:,.0f} m3/d: {overload:.2f}x nameplate; excess={(overload-1)*100:.0f}%")

# The 50% general case
print("Generic 50% liquid increase: Q_new=1.5*Q_base; excess vs unchanged capacity=50%; required incremental capacity=0.5*Q_base.")
