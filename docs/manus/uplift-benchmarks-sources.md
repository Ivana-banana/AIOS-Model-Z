# Журнал проверенных источников

## Проверено 21.08.2026

### Liu, Guthrie, Shipley (2012), SPE-157996-MS — Chevron (объект не раскрыт)
- URL: https://onepetro.org/SPEATCE/proceedings-abstract/12ATCE/All-12ATCE/SPE-157996-MS/156050
- Метод: двухпараметрическая impulse-response / diffusivity-filter модель; Extended Kalman Filter для allocation factors; линейное программирование при ограничениях на доступную воду, ставки и жидкость.
- Факт внедрения: трёхмесячный пилот.
- Реализованный результат: около 22% снижения суммарной суточной закачки; около +2% добычи относительно двухмесячной базовой кривой падения. Относительно 6-месячной кривой добыча соответствовала прогнозному тренду.
- Модельный численный прогноз до пилота: не раскрыт в доступном авторском abstract.
- Число скважин, обводнённость, регион: не раскрыты.

### Jia et al. (2020), Petroleum Exploration and Development, DOI 10.1016/S1876-3804(20)60084-2 — сложный разломный блок, восточный Китай
- URL: https://www.sciencedirect.com/science/article/pii/S1876380420600842
- Метод: сопряжение традиционной численной модели с данным послойной закачки/добычи; data assimilation (history match 85%); ML для оценки эффективности закачки по пластам/направлениям; particle swarm optimization для детального плана.
- Реализованный результат: накопленная добыча за 12 месяцев после оптимизации на 8.2% выше, чем до неё.
- Модельный численный прогноз: не отделён от результата/не найден в доступной странице.
- Число скважин и обводнённость: не раскрыты в открытом abstract; объект прямо назван mature oil field.

### Lu, Xu (2017), SPE-186431-MS — несколько нераскрытых зрелых песчаниковых объектов
- URL: https://onepetro.org/SPEAPOG/proceedings/17APOG/3-17APOG/D031S029R006/194023
- Подтверждённые реализованные примеры: zonal injection после conversion commingled-to-zonal — +11.8% incremental recovery; изменение направления фильтрации модификацией сетки при 98% water cut и 54.5% recovery — +10% incremental recovery; подразделение нагнетательно-добывающей единицы с infill — +9–20% incremental recovery; cyclic injection — +1.84% EUR при сокращении воды/обводнённости.
- Горизонты и число скважин: не раскрыты в abstract; данные являются результатами best practices, не модельным прогнозом.

### Al Saidi, Pourafshary, Al Wadhahi (2015), SPE-172633-MS — Omani oil field
- URL: https://onepetro.org/SPEMEOS/proceedings-abstract/15MEOS/All-15MEOS/182832
- Метод: Capacitance-Resistive Model; ranking injectors, 59% available water sent to most effective wells.
- Обещано моделью: +29.90% future oil production vs current injection profile на следующие 3 года.
- Промысловая реализация: не найдена/не заявлена; результат модельный.
- Масштаб/обводнённость: не раскрыты в abstract.

### Ohms et al. (2010), SPE-121761-PA — BP Alaska, изолированная система 1 нагнетатель + 2 добывающие
- URL: https://onepetro.org/PO/article-abstract/25/03/247/192227/Incremental-Oil-Success-From-Waterflood-Sweep
- Метод: diversion in-depth в канале водонагнетания; дизайн с лабораторными испытаниями, numerical simulation, pressure/chemical-tracer tests.
- Обещано моделью: 50–250 тыс. bbl additional oil за 10 лет.
- Реализовано: более 60 тыс. bbl за первые 4 года; глубинное размещение подтверждено pressure-falloff и injectivity tests.
- Ограничение: специальное конформанс-мероприятие, не чистое перераспределение ставок; микропилот меньше требуемого масштаба, потому в основной выборке его следует держать отдельно/как иллюстрацию разрыва диапазона.

### Alhuthali et al. (2010), SPE-118948-PA — super-giant Middle Eastern field
- URL: https://onepetro.org/REE/article/13/03/406/192160/Field-Applications-of-Waterflood-Optimization-via
- Метод: streamline optimization, equalization waterfront arrival time; stochastic optimization, sequential quadratic programming; rate control smart wells.
- Масштаб: portion of a field with nearly 300 wells, >50 лет истории добычи; 20-year optimisation, facility/rate/voidage constraints.
- Эффект: в открытом summary качественно заявлен considerable oil increase и water decrease; точные числа не найдены; фактическая реализация не заявлена (модельное исследование на реальном объекте).


### Grinestaff, Caffrey (2000), SPE-63152-MS — Northwest Fault Block, Prudhoe Bay, Alaska
- URL: https://onepetro.org/SPEATCE/proceedings-abstract/00ATCE/00ATCE/132224
- Масштаб: history match для 200+ скважин; сложное многопластовое зрелое заводнение.
- Метод и меры: decline curves, Hall plots и review well histories; high-resolution 3D streamline model; управление нагнетателями, остановка расширения заводнения в зоне gravity drainage, замена нагнетателей.
- Реализовано: закачка снижена на 30–40%, заявлены рост добычи и экономия затрат; численная нефтяная прибавка/NPV не раскрыта.
- Подтверждение: history match каждой скважины; inefficiencies позже verified horizontal drilling.

### Feng et al. (2009), SPE-110492-PA — BZ25-1s, Bohai Bay, Китай
- URL: https://onepetro.org/REE/article/12/06/865/191829/Water-Injection-Optimization-for-a-Complex-Fluvial
- Объект: комплексный fluvial heavy-oil (~200 cP), China; короткая история, ранняя стадия (поэтому пограничен по критерию mature).
- Метод и меры: интеграция геологии, ГИС, сейсмики и production data; подбор кандидатов на перевод producer-to-injector, времени и ставок; постепенное увеличение закачки и последующая корректировка injection/production schedules.
- Реализованный эффект через 1 год: decline oil rate на platform D 26%→19%, E 47%→14%; давление стабилизировано. Численная дополнительная нефть/NPV не раскрыта.

### Shchipanov et al. (2008), SPE-116873-MS — North Sea heterogeneous sandstone
- URL: https://onepetro.org/SPERPTC/proceedings-abstract/08ROGC/08ROGC/SPE-116873-MS/145195
- Метод: field-history analysis, analytical prescreen и sector numerical model; циклическая закачка/переключение patterns.
- Прогноз модели: рост добычи до 3% для коротких и 5% для длинных циклов с сокращением воды.
- Факт: анализ реальной цикличности injection/production в поле; отдельный численный фактический прирост из открытого abstract не найден.

### Khan et al. (2014), SPE-171777-MS — giant UAE offshore carbonate
- URL: https://onepetro.org/SPEADIP/proceedings-abstract/14ADIP/14ADIP/210644
- Метод: history-matched finite-difference + streamlines; WAF, pair injection efficiency и VRR; одновременно повысить producer rates для high-IE pairs, снизить для low-IE и переназначить ставки нагнетателей.
- Результат: описан как прогнозная оптимизация (higher oil or lower produced/injected water); числа, фактическая реализация, WCT и масштаб не раскрыты в abstract.

### Guerrero-Lee et al. (2022), SPE-209281-MS — Belridge Diatomite Waterflood, California
- URL: https://onepetro.org/SPEWRM/proceedings-abstract/22WRM/22WRM/484146
- Масштаб: во всём BDWF 1,900+ injectors и 2,600+ producers; пилот West Grande: 52 injection strings, 142 producers, 10,500 STB/d injection, 8,200 STB/d water, 700 STB/d oil.
- Метод/меры: streamline surveillance, целевые ставки с ограничениями по Pwh, surface elevation, communication, pump runtime/capacity; ставки меняли 4 раза за 9 месяцев.
- Промысловый результат: reversal oil decline по сравнению с предыдущим годом; авторы отдельно предупреждают о неопределённостях замеров добычи/WCT и interference from operations. Численный прирост/NPV не раскрыт.

### Zhang et al. (2024 online / 2025 issue), DOI 10.1115/1.4066279 — offshore unconsolidated sandstone (оператор CNOOC; объект не назван)
- URL: https://asmedigitalcollection.asme.org/energyresourcessubsurface/article/1/1/011001/1203079/Optimization-of-Injection-Production-Relationship
- Объект: offshore unconsolidated sandstone, high water-cut stage; масштаб и регион не раскрыты.
- Метод: multilayer inter-well connectivity network + automatic history matching + real-time injection-production optimization.
- Реализовано: +16,000 м³ нефти за год и снижение обводнённости на 0.87 п.п.; модельный процент/NPV не раскрыты.

### Li et al. (2021), Petroleum Exploration and Development — Middle East carbonate, linear horizontal pattern
- URL: https://www.sciencedirect.com/science/article/pii/S1876380421600976
- Объект: высокообводнённый carbonate Middle East; test well group, размер не раскрыт.
- Метод/меры: превращение linear injection в cyclic alternating patterns, оптимизация injection/production parameters, изменение направления вытеснения.
- Реализовано: максимум +23.84 м³/сут на скважину и максимум −18 п.п. water cut. Горизонт и модельный прогноз не раскрыты.

### Agudelo et al. (2016), SPE-180774-MS — Casabe, Colombia
- URL: https://onepetro.org/SPETTCE/proceedings-abstract/16TTCE/16TTCE/188279
- Объект: Casabe, Colombia; водозаводнение с 1985, mature; последние 2 года рост water cut, ускоренное падение и нестабильность; коллектор гетерогенный, с плохой песчаной связностью/неблагоприятной подвижностью.
- Метод/меры: детальная surveillance-driven optimization; закачка в пласты с высоким remaining oil и прекращение закачки в высокообводнённые; pressure monitoring, PLT, меры по sediment production.
- Результат: численное значение не раскрыто в открытом abstract; authors characterize implementation as successful management/increased production в публикационном описании.

### Choudhary et al. (2014), SPE-169132-MS — Ebano, offshore Equatorial Guinea
- URL: https://onepetro.org/SPEIOR/proceedings-abstract/14IOR/All-14IOR/SPE-169132-MS/211191
- Объект: high-permeability slope-channel turbidite; на момент статьи 2 wells (1 injector, 1 producer), расстояние 1.5 км; нефть с мая 2009, injection с июля 2009.
- Существенное расхождение с прогнозом: фактический water breakthrough примерно через 1 год, существенно раньше первоначального прогноза. Причина авторов: вероятные thief zones водонагнетания.
- Меры: TAP in-depth conformance pilot, April 2011; 48,000 bbl at 6,000 bbl/d, 15,000 ppm. Подтверждения: injection/production data, pressure, produced-water chemistry, PLT, fall-off tests; PLT показал неизменность профиля, остаются uncertainties in activation distance, pressure response (simulation vs field), production performance. Численно «обещано/получено» по нефти не раскрыто.

### Li et al. (1997), SPE-30875-PA — Southern oilfields, Daqing Placanticline
- URL: https://onepetro.org/ATS/article/5/01/20/107903
- Метод: циклическое заводнение, сопоставление field case histories traditional vs cyclic at different WCT.
- Результат: качественно подтверждены control of water-cut rise, higher oil production и higher injection efficiency; exact numeric oil/NPV, time horizon и well count не раскрыты.

### Nolen-Hoeksema et al. (1994), SPE-24873-PA — McElroy и North Westbrook Unit, Permian Basin
- URL: https://onepetro.org/RE/article/9/04/254/69923/Waterflood-Improvement-in-the-Permian-Basin-Impact
- Меры: waterflood pattern realignment на основе оценки in-situ stress/fracture direction.
- Реализованный результат по McElroy field, по авторскому поисковому описанию OnePetro: +40% field oil rate, 13,000→18,000 BOPD; in-page abstract подтверждает successful realignment, но горизонт, WCT, well count, forecast/NPV не раскрыты. В финальном отчёте привязать количественное значение к ссылке OnePetro с пометкой «указано в индексируемом описании страницы; полный текст платный».

### Lemen et al. (1990), SPE-20120-MS — McElroy Field Section 205, Permian Basin
- URL: https://onepetro.org/SPEPBOGR/proceedings-abstract/90OGR/All-90OGR/68595
- Объект/меры: 640 acres (259 ha), realignment 1988; irregular/widely spaced patterns→smaller uniform patterns.
- Результат: economic viability and improved waterflood performance; oil/NPV/WCT/horizon not disclosed in abstract.

### Ghori et al. (2006), SPE-93031-MS — Saudi Arabian carbonate
- URL: https://onepetro.org/SPEIOR/proceedings-abstract/06IOR/All-06IOR/140711
- Метод: 3D streamline, WAF/attribution; modification of injection allocation and locations through simulations; emphasis is incremental development with horizontal injection/production wells.
- Итог: improved injection pattern model result; no field response, WCT, size, or numbers disclosed.

