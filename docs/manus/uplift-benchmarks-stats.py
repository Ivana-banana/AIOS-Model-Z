from statistics import median
from pathlib import Path
import csv

base = Path('/home/ubuntu/waterflood_cases')

# Only values with the same published metric and numerator/denominator are pooled.
# All values are directly transcribed from the cited sources.
groups = {
    'realized_incremental_recovery_pct': {
        'label': 'Реализованный прирост извлечения, %',
        'values': [11.8, 10.0, 1.84],
        'source_cases': ['Lu & Xu 2017: zonal injection', 'Lu & Xu 2017: flow-direction change', 'Lu & Xu 2017: cyclic injection'],
    },
    'realized_relative_oil_rate_pct': {
        'label': 'Реализованный относительный прирост дебита нефти, %',
        'values': [2.0],
        'source_cases': ['Liu et al. 2012: 3-month pilot vs 2-month decline curve'],
    },
    'modelled_npv_pct': {
        'label': 'Модельный прирост NPV, %',
        'values': [400.0],
        'source_cases': ['Halliburton Consulting case study: vs operator existing plan'],
    },
}

with (base / 'cases.csv').open(newline='', encoding='utf-8') as f:
    cases_count = sum(1 for _ in csv.DictReader(f))

out_lines = [
    '# Расчёт статистики',
    '',
    f'Число строк-кейсов в реестре: {cases_count}.',
    '',
    'В расчёт включены только численно раскрытые проценты в одной и той же метрике. ' \
    'Показатели дебита, накопленной нефти, дополнительного КИН, водоотдачи и NPV не объединяются.',
    ''
]

for key, group in groups.items():
    values = sorted(group['values'])
    out_lines += [
        f'## {group["label"]}',
        '',
        f'- n: {len(values)}',
        f'- минимум: {values[0]:.2f}%',
        f'- медиана: {median(values):.2f}%',
        f'- максимум: {values[-1]:.2f}%',
        '- кейсы: ' + '; '.join(group['source_cases']),
        ''
    ]

out_lines += [
    '## Неагрегируемые показатели',
    '',
    '- Реализованная накопленная нефть за 12 месяцев: +8.2% (один кейс; отдельная база сравнения).',
    '- Реализованные абсолютные эффекты: >60 Mbbl за 4 года; +16,000 м³/год; до +23.84 м³/сут на скважину. Они не переводятся в проценты.',
    '- Модельные нефтяные эффекты: +29.90% future oil на 3 года; до +3% / +5% при циклических режимах; 50–250 Mbbl за 10 лет. Разные метрики и горизонты, единую медиану считать нельзя.',
    '- Полный промысловый показатель +40% на McElroy упомянут в индексируемом описании страницы OnePetro, но не повторяется в открытом abstract; поэтому он приведён в таблице с оговоркой, но исключён из статистики строгой выборки.\n- Реализованный NPV: не найдено в 23 кейсах.',
    '- Отношение «факт/модель» для нефти: не рассчитано; не найдено ни одного кейса с численно раскрытыми модельным и промысловым нефтяными эффектами в одной единице и на одном горизонте.',
]

(base / 'stats.md').write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
print('\n'.join(out_lines))
