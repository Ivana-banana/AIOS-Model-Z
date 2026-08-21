# CRM и proxy-модели пласта для заводнения

**Автор:** Manus AI  
**Дата:** 21 августа 2026 г.  
**Задача сравнения:** 103 скважины, 370 месяцев истории, 48 обучающих сценариев.

## Краткий вывод

Для набора с **103 скважинами и длинной помесячной историей** наиболее обоснованная первая линия — **разреженный CRMP** с физическими ограничениями, а не полный CRMIP. CRMP даёт по одному времени отклика на добывающую скважину и матрицу связности «нагнетатель → добывающая», поэтому хорошо подходит для диагностики, history match и базовой оптимизации закачки. CRMIP стоит применять лишь после геологического/геометрического отсечения пар: без отсечения число параметров растёт приблизительно как $N_{inj}N_{prod}$ и при разбиении 50/53 достигает 10 600 параметров в типичной полной параметризации.

**48 независимых сценариев** — это мало для обучения «с нуля» полноразмерного FNO, DeepONet или GNN-оператора, если ожидается обобщение на новые геологии и новые режимы управления. В работах по нейросетевым proxy обычно используются сотни сценариев: 300 для E2C-ROM, 450 для GNN–LSTM и 800 для FNO [13] [14] [15] [17]. При этом 370 месяцев создают богатую временную выборку, но не заменяют разнообразие **независимых** управляющих и геологических траекторий.

Практически разумная архитектура — **CRM как физический backbone + ML на остатке**: CRM оценивает $f_{ij}$ и $\tau_j$, его прогноз и импульсные отклики становятся признаками, а регуляризованная сеть корректирует нелинейность двухфазного потока, water cut и эффекты режимов. Если нужны пространственные поля давления/насыщенности, то следует строить simulator-trained proxy, но для 48 сценариев использовать multi-fidelity/transfer learning, а граф и связи инициализировать CRM/геологией, а не обучать свободную плотную сеть.

> **Интерпретация числа 48.** Ниже предполагается, что это 48 независимых сценариев управления/геологии, каждый из которых может содержать до 370 месячных шагов. Если это 48 *наблюдений*, а не сценариев, нейросетевые proxy практически неидентифицируемы; если это 48 полных траекторий одного фиксированного пласта, можно обучать компактный hybrid или GNN, но нельзя ожидать широкого out-of-distribution обобщения.

| Вопрос | Рекомендация для 103 / 370 / 48 |
|---|---|
| Диагностика межскважинной связи и перераспределение закачки | Разреженный **CRMP** с ограничениями $f_{ij}\ge0$, $\sum_j f_{ij}\le1$, $\tau_j>0$; fitting windows по полевым событиям. |
| Нелинейность oil/water rates, водообводнённость | **CRM + residual ML** либо CRM-PINN: прогноз CRM, его residual, BHP и water-cut признаки подаются в небольшую сеть. |
| Отдельный $\tau$ на каждую пару | CRMIP — только на candidate edges после отсечения по расстоянию, общему пласту/перфорации, трассерам, streamline или первичному CRMP. |
| Быстрый proxy по дебитам под новыми режимами | Графовый CRMP-informed GNN–LSTM/GRU с малым числом параметров; 48 сценариев — pilot, не production-grade universal surrogate. |
| Поля $p,S_w$ на сетке | POD-ROM/POD-DEIM — наиболее data-efficient при доступе к симулятору; FNO/DeepONet — после расширения библиотеки сценариев примерно до сотен или через transfer/multi-fidelity. |

---

## 1. Что именно моделирует CRM

CRM — это семейство lumped material-balance моделей, в котором система «нагнетатель–пласт–добывающая скважина» представляется аналогом сопротивления и ёмкости. Модель использует историю закачки, добычи и, когда они доступны и сопоставимы по datum, BHP добывающих скважин. Ей не нужна полноценная геологическая модель или петрофизика для самой калибровки, но эти данные критичны для **интерпретации** полученной связности и отбора допустимых пар [6].

Базовый вывод строится из материального баланса и линейной продуктивности:

$$c_tV_p\frac{d\bar p}{dt}=w-q,\qquad q=J(\bar p-p_{wf}),\qquad \tau=\frac{c_tV_p}{J}.$$

После исключения среднего пластового давления получается базовая ОДУ:

$$\tau\frac{dq}{dt}+q=w-\tau J\frac{dp_{wf}}{dt}.$$ 

Здесь $q$ — жидкостная добыча в пластовых объёмах, $w$ — закачка, $J$ — продуктивность, $c_tV_p$ — эффективная ёмкость контрольного объёма, а $\tau$ — постоянная времени. Величина $1/\tau$ интерпретируется как характерный темп затухания/отклика; строгое физическое сопоставление с объёмом и проницаемостью требует выполнения исходных допущений [3] [6].

Классическое CRM предполагает постоянную температуру, слабую сжимаемость, пренебрежимый капиллярный эффект, постоянный объём контрольной области, мгновенное выравнивание давления внутри неё и постоянный $J$. Поэтому это модель отклика **среднего** контроля, а не полносеточный симулятор фронта вытеснения [3] [6]. После прорыва воды параметры часто рассматривают как квазипостоянные; до прорыва и при сильных изменениях насыщенности данное допущение хуже выполняется.

### 1.1. CRMT, CRMP и CRMIP

| Вариант | Контрольный объём и уравнение | Оцениваемые параметры | Когда использовать | Главная цена упрощения |
|---|---|---|---|---|
| **CRMT** | Один tank на весь пласт/сектор: $\tau\dot q+q=f w-\tau J\dot p_{wf}$. Здесь $q,w$ — суммарные псевдодобыча и псевдозакачка. | $f,\tau,J,q_0$ по сектору. | Быстрый field/sector history match, проверка voidage, стартовые значения для более детальной модели. | Не даёт индивидуальную связность и дебиты скважин. |
| **CRMP** | Один объём на producer $j$: $\tau_j\dot q_j+q_j=\sum_i f_{ij}w_i-\tau_jJ_j\dot p_{wf,j}$. | $f_{ij}$ для пар и один $\tau_j$ на добывающую; при BHP — $J_j$. | Основной вариант для pattern waterflood и разреженной карты связности. | У всех инжекторов, влияющих на $j$, одинаковый $\tau_j$; плохо при сильной локальной неоднородности/разных расстояниях. |
| **CRMIP** | Отдельный объём на паре: $\tau_{ij}\dot q_{ij}+q_{ij}=f_{ij}w_i-\tau_{ij}J_{ij}\dot p_{wf,j}$, $q_j=\sum_iq_{ij}$. | $f_{ij},\tau_{ij},J_{ij},q_{ij,0}$ по каждой паре. | Нужны разные времена отклика от разных инжекторов к одному producer; малое число кандидатных пар. | Резкий рост размерности и неединственность history match. |

CRMT использует коэффициент $f$ для учёта неполного эффективного использования закачки: $f<1$ может отражать потери в thief zone, ниже WOC, в аквифер либо в соседний объект; $f>1$ в tank-интерпретации может отражать внешнюю поддержку. В multiwell CRMP/CRMIP $f_{ij}$ — это доля приращения закачки инжектора $i$, приходящаяся на изменение добычи в $j$ в установившейся интерпретации [3] [6].

Для дискретных месячных данных и кусочно-постоянной закачки с линейным изменением BHP практична рекурсия CRMP:

$$q_{j,k}=a_jq_{j,k-1}+(1-a_j)\left[\sum_i f_{ij}w_{i,k}-\tau_jJ_j\frac{p_{wf,j,k}-p_{wf,j,k-1}}{\Delta t_k}\right],\qquad a_j=e^{-\Delta t_k/\tau_j}.$$

Она показывает, что CRM — свёрточная модель с экспоненциальной памятью. При отсутствии надёжного BHP последний член обычно исключают, но тогда эффект смены штуцера, ESP, skin или режима BHP может ошибочно приписываться $f_{ij}$ или $\tau$.

### 1.2. Как оцениваются $f$ и $\tau$

Калибровка — это constrained nonlinear least squares по истории жидкостной добычи, либо совместно по $q_L$, $q_o$, $q_w$, BHP и дополнительным данным. Типовой функционал:

$$\min_\theta\;\sum_{j,k}\omega_{j,k}\left(q^{obs}_{j,k}-q^{CRM}_{j,k}(\theta)\right)^2+\lambda_f\Omega_f+\lambda_\tau\Omega_\tau,$$

где $\theta=\{f_{ij},\tau_j,J_j,q_{0,j}\}$ для CRMP или $\theta=\{f_{ij},\tau_{ij},J_{ij},q_{ij,0}\}$ для CRMIP. Вес $\omega$ нужен для нормировки больших и малых скважин, обработки разной достоверности замеров и, при необходимости, акцента на недавнем окне. Базовые работы формулируют CRM как нелинейную многомерную регрессию/сигнальную систему [1] [3].

На практике применяют SLSQP/SQP, interior-point, bounded trust-region или L-BFGS-B с параметризацией, автоматически соблюдающей ограничения. В реализации CRMP из сравнительной работы Sen параметры обучались по scaled MSE с SLSQP, неотрицательными bounds и балансовым ограничением на gains [7]. Важнее выбора оптимизатора — несколько стартов, проверка чувствительности, bootstrap по времени/скважинам и out-of-window forecast: параметр $\tau$ даёт неconvex objective и может существенно зависеть от начального приближения [7].

**Идентифицируемость определяется возбуждением входа.** Если два нагнетателя меняют закачку синхронно, история не позволяет устойчиво разделить их $f_{ij}$, даже если RMSE мала. Полезны несинхронные ступенчатые изменения, остановки и естественные колебания режимов; опасны длинные куски почти постоянной закачки. Историческое окно нужно сегментировать по крупным работам на скважинах, переводу на другой горизонт, GOR/water-cut изменениям и сменам BHP. В полевом 99-well примере fitting window выбирался именно с учётом событий, затем выполнялась backtest-проверка на оставшейся истории [4].

### 1.3. Физические ограничения на коэффициенты

Если $f_{ij}$ определён как доля потока **из инжектора $i$ в добывающую $j$**, обязательное балансовое ограничение имеет направление суммы по producers:

$$f_{ij}\ge0,\qquad \sum_{j=1}^{N_{prod}}f_{ij}\le1,\qquad \tau_j>0\;\text{или}\;\tau_{ij}>0,\qquad J\ge0.$$

Равенство $\sum_jf_{ij}=1$ оправдано только тогда, когда в модели представлены все пути от данного инжектора и принято отсутствие внешнего sink/source. Неравенство строго предпочтительнее при подозрении на thief zones, уход под WOC, перетоки между объектами или неполный spatial window [6]. При другой договорённости, когда коэффициенты нормируют как доли **входов в producer**, сумма берётся по $i$; смешивать эти соглашения нельзя.

| Ограничение | Зачем оно нужно | Практическая реализация |
|---|---|---|
| $f_{ij}\ge0$ | Отрицательная «связь» физически не является долей закачки; часто это симптом коллинеарности сигналов, пропущенного BHP/события или неверной области. | $f_{ij}=s_i\,\mathrm{softmax}_j(a_{ij})$ либо $f_{ij}=\mathrm{softplus}(a_{ij})$ с линейным inequality. |
| $\sum_j f_{ij}\le1$ | Материальный баланс одной закачки; остаток — неучтённый выход. | $s_i=\sigma(b_i)$ и softmax по $j$: сумма автоматически равна $s_i\le1$. |
| $\tau>0$, $J\ge0$ | Исключает неустойчивый экспоненциальный отклик и нефизичную отрицательную продуктивность. | $\tau=\tau_{min}+\mathrm{softplus}(b)$; $J=\mathrm{softplus}(c)$. |
| Sparsity/locality | Снижает число параметров и ложные дальние связи. | Заранее оставить edges в радиусе/общем слое; group-lasso, $L_1$ на $f$, либо mask из геологии/CRM. |
| Гладкость во времени | Не выдавать шум за динамическую связность. | Piecewise-constant $f(t)$ по событиям или penalty $\sum_t\|f_t-f_{t-1}\|^2$. |

Сумма $f_{ij}$ физически близка к доле межскважинной проводимости только в ограниченном режиме: зрелый pattern flood, приблизительно стабильные BHP и закачка, отсутствие сильной смены структуры потока. Поэтому карту CRM следует проверять по расстоянию, перфорациям, layer overlap, PLT, трассерам, 4D seismic/streamlines и оперативной истории, а не трактовать как единственную геологическую истину [6].

### 1.4. Какие данные нужны

Минимальная согласованная таблица должна содержать даты, статусы скважин, объёмы закачки по каждому инжектору и жидкостную добычу по каждому producer, приведённые к одинаковым условиям и временному шагу. Месячная частота допустима для медленных водонапорных систем, но быстрые отклики, короткие shut-in и переходные режимы лучше агрегировать аккуратно либо анализировать на более частом шаге.

| Блок данных | Нужен для | Критические проверки |
|---|---|---|
| $w_i(t)$: закачка воды | Входы CRM и всех proxy. | Пластовые/поверхностные условия, работающие дни, пропуски, инверсия знака, список активных инжекторов. |
| $q_{L,j}(t)$, $q_{o,j}(t)$, $q_{w,j}(t)$, water cut | Target CRM и ML; связь с fractional-flow моделью. | Баланс $q_L\approx q_o+q_w$, корректность нулей и stopped wells. |
| $p_{wf,j}(t)$, choke, ESP/газлифт, skin-индикаторы | Отделяет производительность скважины от межскважинного отклика. | Общий datum, частота, калибровка манометров, отметка смены режима. |
| Координаты, траектории, перфорации, пласты, фации, $k,h,\phi$ | Mask кандидатных пар, prior для $f$, графовые ребра, spatial proxy. | Взаимодействие только между общими интервалами, горизонтальная длина, расстояние по пласту. |
| События и ограничения | Segmentation/causal filtering, безопасная оптимизация. | Ремонты, ГТМ, закрытие, ограничения трубопроводов/ВПП, лимиты закачки и воды. |
| PLT, tracer, interference/pressure transient, 4D | Внешняя валидация связности. | Разная временная шкала и неопределённость; нельзя бездумно смешивать с rate data. |

CRM по умолчанию прогнозирует **жидкость**. Прогноз $q_o$ и $q_w$ требует либо отдельной модели fractional flow/water cut, либо двухфазного расширения. Связка CRM с fractional flow используется именно для перехода от liquid-rate к oil/water-rate и для оптимизации закачки [3] [6].

---

## 2. Как сочетать CRM и ML

### 2.1. Три уровня гибридизации

**Первый уровень — ML как более гибкий прогнозатор.** RNN/LSTM/TCN/Transformer получает историю закачки, собственную добычу и BHP, а затем прогнозирует $q_L$, $q_o$, $q_w$ или water cut. В сравнении CRMP и простой RNN веса матрицы входов RNN оказались качественным аналогом карты connectivity, а recurrent weights — аналогом памяти системы; RNN лучше воспроизводила нелинейные синтетические отклики, но требовала больше времени обучения [7]. Это полезно, но веса сети без ограничений нельзя автоматически объявлять физическими $f_{ij}$.

**Второй уровень — residual hybrid.** Сначала строится CRM, затем ML моделирует остаток:

$$\hat q_j(t)=q^{CRM}_j(t;\hat f,\hat\tau,\hat J)+g_\phi\big(x_j(t),\;q^{CRM}_j(t),\;r^{CRM}_j(t-1)\big).$$

Такой дизайн оставляет CRM интерпретируемым и позволяет сети учить главным образом двухфазную нелинейность, эффекты работы скважины и систематические пропуски. Входные признаки $q^{CRM}$, $f_{ij}$, $\tau_j$, свёртки $f_{ij}w_i$ и lagged CRM residual особенно ценны при малом числе сценариев.

**Третий уровень — physics-informed CRM network.** В CRM-PINN в loss одновременно входят невязка по данным и остаток CRM-ОДУ; в ходе backpropagation обновляются веса сети и физические параметры, включая connectivity factors. В работе на синтетическом и зрелом реальном объекте CRM-regularization улучшала предсказательность по сравнению с чистой ANN при более сложном поведении пласта [8].

$$\mathcal L=\mathcal L_{data}(q_o,q_w,q_L)+\lambda_{CRM}\sum_{i,j,t}\left\|\tau_{ij}\dot q_{ij}+q_{ij}-f_{ij}w_i+\tau_{ij}J_{ij}\dot p_{wf,j}\right\|^2+\lambda_{MB}\mathcal L_{balance}+\lambda_{reg}\mathcal L_{sparse}.$$

Для целевого набора это предпочтительнее «чёрного ящика»: параметры можно ограничить softmax/softplus, а по $f$ и $\tau$ получить диагностику. Важно не называть такую модель полноценным PINN PDE-solver: физика в loss — **сведённая CRM-динамика**, а не полносеточные PDE двухфазной фильтрации.

### 2.2. Рекомендуемые признаки для сети

| Группа | Примеры признаков | Роль |
|---|---|---|
| Управление нагнетателями | $w_i(t)$, $\Delta w_i$, rolling sums/means, cumulative injection, shut-in flag, VRR, time since rate change. | Создаёт причинный входной сигнал. |
| Динамика producer | Lags $q_L,q_o,q_w$, water cut, BHP, $\Delta p_{wf}$, choke/ESP/газлифт, uptime, time since workover. | Разделяет пластовый ответ и работу скважины. |
| CRM-представление | $q^{CRM}$, residual, $f_{ij}$, $\tau_j$, $\sum_if_{ij}w_i$, экспоненциальные свёртки по каждому инжектору. | Физический prior и компактная память. |
| Статические скважинные/геологические | Координаты, расстояние и азимут, layer overlap, перфорации, $k,h,\phi$, fault/channel flags, completion type. | Маска и edge features для GNN; препятствует ложным дальним связям. |
| Контекст времени | Месяц/сезонность только при наличии физического смысла, cumulative oil/water, время от water breakthrough, regime ID. | Показывает стадию выработки. |

Если target — добыча следующего месяца, разделение train/validation/test должно быть **по времени и по сценариям**, а не случайным перемешиванием точек. В противном случае модель видит почти идентичные соседние месяцы и получает завышенную оценку качества. Для history-matching добавляют rolling-origin backtest: fit на $[0,T_1]$, forecast на $(T_1,T_2]$, повторяют для нескольких $T_1$.

### 2.3. CRM-informed графовая сеть

Для 103 скважин естественен directed bipartite graph: нагнетатели — source nodes, producers — target nodes, ребро $i\to j$ существует только в physical candidate set. Его edge features: расстояние, общий интервал, layer overlap, $\hat f_{ij}$, $\hat\tau_j$ или $\hat\tau_{ij}$, статическая проводимость/streamline prior. Временные node features — rates, BHP, water cut, uptime и управляющие изменения. GNN/GAT выполняет message passing по связям, LSTM/GRU/TCN — эволюцию во времени.

Это прямо согласуется с GNN–LSTM, где граф кодирует injector–producer и producer–producer связи, а temporal block прогнозирует oil и water rates [15]. Другой вариант GCN–GRU использует граф, начатый из $k,\phi$, расстояния и динамических данных, чтобы корректировать interwell connectivity, а затем прогнозировать oil rate и water cut [16]. Для 48 сценариев важна **жёсткая разреженность**: полный ориентированный граф на всех скважинах приводит к слишком большому числу свободных весов и переобучению.

---

## 3. Proxy-модели пласта

### 3.1. POD-ROM, POD-TPWL и POD-DEIM

В projection-ROM состояние full-order модели $x(t)=[p(t),S_w(t),\ldots]$ аппроксимируется низкоранговым базисом snapshots:

$$x(t)\approx \bar x+\Phi_r z(t),\qquad X=[x(t_1),\ldots,x(t_m)]=U\Sigma V^\top,$$

где $\Phi_r$ составлен из первых $r$ левых сингулярных векторов. Затем динамику $z(t)$ получают Galerkin-проекцией, TPWL-линеаризацией вдоль траекторий или DEIM-аппроксимацией нелинейных членов. POD-DEIM особенно полезен при доступе к внутренностям симулятора; E2C/POD-TPWL-подобные подходы могут быть non-intrusive и учиться по входам/выходам симулятора [10] [13] [14].

POD-ROM обычно требует **меньше независимых сценариев**, чем нейрооператор, если целевая область контролей узка: snapshots нескольких representative trajectories можно адаптивно расширять после валидационного full-order прогона. В базовой работе 4 050 полных состояний были сжаты до 20–100 ROM-состояний, с близкими к full-order оптимальными стратегиями и максимум 35% экономии времени; это иллюстрация, а не универсальная гарантия [10]. POD-DEIM в моделях заводнения также использует snapshots по длительной траектории и сохраняет консервативность вспомогательного velocity field [11].

Для 48 сценариев POD-ROM — реальный кандидат, **если имеется симулятор и состояния $p,S_w$**, а сценарии покрывают рабочий диапазон. Если доступны только rates, но не сеточные states, лучше CRM/hybrid или E2C-type latent proxy.

### 3.2. DeepONet

DeepONet учит оператор $\mathcal G:u\mapsto y$. Branch network принимает функцию-вход: поле $k,\phi$, карту скважин и/или дискретизированный schedule контролей. Trunk network принимает точку запроса $(x,y,z,t)$. Скалярный выход строится как произведение latent-кодов branch и trunk; таким образом один оператор может выдавать поле в произвольных пространственно-временных query points. Для задач пласта target — $p(x,t)$, $S_w(x,t)$, $S_g(x,t)$ или well QoI.

U-DeepONet для CO$_2$–water использует U-Net branch для полевых и скалярных входов и trunk по времени; в опубликованном эксперименте входной тензор имеет пространственный размер 96×200, 24 временных точки и 12 каналов, а отдельные сети прогнозируют saturation и pressure buildup [18]. Это близко по классу задачи, но не тождественно нефтяному заводнению. Его плюс — query-based выход и разделение пространственных признаков/времени; минус — обучающая библиотека должна покрывать совместное разнообразие геологии, положений скважин и control schedules.

### 3.3. Fourier Neural Operator

FNO заменяет локальные свёртки операторными слоями в пространстве Фурье и учит отображение между полями. Для structured grid можно подавать каналы $[k,\phi,\text{well/source map},\text{control},t]$ и предсказывать $p$ или $S_w$. Его сильная сторона — глобальная рецептивная область и возможность перехода на другую сеточную дискретизацию; слабая — резкие градиенты у скважин и фронта вытеснения часто сглаживаются.

В waterflood FNO-кейсе Ma et al. использовали 1 000 геостатистических реализаций, одну пару injector–producer, 241×241 сетку, 720 дней и 48 snapshots; 800 реализаций обучали, 200 тестировали. Входами были $k$, $\phi$ и время, target — поле давления; метрики — RMSE, MAE и MSE. Авторы отдельно показывают, что zero-shot super-resolution с 241×241 до 1205×1205 ухудшается около скважин, а multi-fidelity refinement помогает [17]. Следовательно, **48 сценариев не следует сопоставлять с 48 time snapshots**: в этой работе независимых входных реализаций было 1 000.

### 3.4. Graph neural networks по связности скважин

GNN наиболее естественна, когда output — well rates, water cut или connectivity, а не полное сеточное поле. Она масштабируется по числу активных ребер, допускает переменное число скважин и использует доступные для инженера priors. Нужны явные decision rules: направленные ребра $i\to j$, masked attention, ограничение $\sum_j f_{ij}\le1$ для интерпретируемого connectivity head и separate temporal state.

Крупный GNN–LSTM кейс обучал 450 из 500 high-fidelity симуляций на объекте из 189 скважин; средние относительные ошибки для water и oil rates составили около 4.8% и 4.3%, а online speedup превысил 1 000× [15]. Это наиболее релевантный масштаб по числу скважин, но также прямое свидетельство, что 48 сценариев недостаточны для свободного graph surrogate без сильных priors.

### 3.5. Типичные объёмы данных и метрики

| Семейство | Что является обучающей единицей | Порядок объёма в найденных работах | Метрики, которые следует приводить | Вывод для 48 сценариев |
|---|---|---|---|---|
| CRMP/CRMIP | Месячные ряды на скважины и вмешательства. | Один реальный объект в 99 скважин [4]; малые синтетические кейсы [7]. | Weighted RMSE/MAE на $q_L$, NRMSE; rolling forecast; ошибки cumulative oil/water; карта $f$ против независимых данных. | **Да**, при сигнале в закачке и разрежении пар. |
| POD-ROM/POD-DEIM | Snapshots full-order state из representative trajectories. | Десятки/сотни snapshots; 4 050→20–100 состояний в [10]; длительная 1 000-day траектория в POD-DEIM [11]. | Relative $L_2$ по $p,S_w$; well-rate RMSE; cumulative oil/NVP error; water-breakthrough error; speedup. | **Да**, если есть симулятор и states; active enrichment обязателен. |
| E2C/physics-aware proxy | Independent control schedules из симулятора. | 300 runs, 9 wells [13]; 300 control sets, 9 wells [14]. | State-field errors + QoI RMSE/MAE/MAPE, mass residual, speedup, OOD-control error. | **Погранично**: компактная сеть и physics loss; 48 заметно меньше 300. |
| GNN–LSTM/GRU | Сценарии + граф + временные well attributes. | 450 train scenarios, 189 wells [15]; пять reservoir cases в [16]. | Mean relative error/MAPE, RMSE/MAE по well rates/water cut, connectivity comparison, cumulative error. | **Пилот**, если CRM/геология задаёт sparse graph; не свободная dense GNN. |
| FNO | Независимые field/control realizations, часто с time snapshots как targets. | 800 train из 1 000 realizations [17]. | Field RMSE/MAE/MSE; spectral/relative $L_2$; error near wells/front; super-resolution and speedup. | **Нет с нуля**; нужны аугментация, multi-fidelity или transfer. |
| DeepONet | Функции-входы + пространственно-временные query points. | Эксперимент [18] использует 96×200×24×12 input structure; объём realizations в доступном фрагменте не раскрыт. | Relative $L_2$, MAE/RMSE fields and QoIs, time-interpolation/extrapolation error, training cost. | **Скорее нет с нуля**; возможен low-dimensional/transfer вариант. |

---

## 4. Метрики и валидация, которые должны быть в проекте

Для CRM недостаточно одной training RMSE: модель может точно сгладить прошлую добычу, но неверно распределить влияние инжекторов. Минимальный отчёт должен отделять accuracy rates от reliability connectivity.

| Уровень | Метрики | Как интерпретировать |
|---|---|---|
| Скважинные rates | RMSE, MAE, NRMSE, MAPE/sMAPE; $R^2$ с осторожностью. | Считать отдельно для $q_L,q_o,q_w$; MAPE неустойчив при почти нулевых дебитах. |
| Накопленные показатели | Ошибка cumulative oil/water/injection, NPV или proxy objective. | Важнее pointwise RMSE при оптимизации закачки. |
| Динамика | Ошибка времени water breakthrough, cross-correlation lag, error в response на step change. | Проверяет именно $\tau$ и причинный отклик. |
| Поля proxy | Relative $L_2$, MAE/RMSE по $p,S_w$, mass-balance residual, error вокруг скважин/фронта. | Средняя ошибка поля не должна скрывать провал у скважин. |
| Связность | Precision/recall top-$k$ edges на synthetic truth; ранговая корреляция с tracer/streamline/PLT; устойчивость при bootstrap. | Для real field обычно нет истинного $f$; нужна внешняя и экспертная валидация. |
| Обобщение | Forecast на будущих месяцах; leave-scenarios-out; OOD controls; stress-tests (shut-in, изменение лимитов). | Random split по месяцам недопустим как основной тест. |
| Стоимость | Offline simulator cost, training time, online latency, speedup относительно full physics. | Отделять время построения от времени одного прогноза. |

---

## 5. Конкретный дизайн для 103 скважин, 370 месяцев, 48 сценариев

### Шаг 1. Сначала сделать CRM-базу

Разделите скважины на инжекторы и producers и сформируйте candidate graph по общему объекту/пласту, перекрытию перфораций и расстоянию. Даже без геомодели нельзя автоматически создавать все пары. Если предположить примерно 50 инжекторов и 53 producers, полный граф содержит 2 650 пар. Для CRMP это около 2 756 параметров только для $f_{ij}$ плюс одного $\tau_j$ и $J_j$ на producer; для полной CRMIP при четырёх параметрах на пару получается около 10 600 параметров. Поэтому на первом цикле оставьте, например, $K=5$–10 candidate injectors на producer и проверяйте чувствительность к $K$.

Оцените constrained CRMP на нескольких fitting windows, с $f_{ij}\ge0$, $\sum_jf_{ij}\le1$, $\tau_j>0$ и, когда BHP достаточен, $J_j\ge0$. Сохраняйте не только best fit, но также bootstrap distribution $f_{ij}$ и стабильность top-$k$ edges. Постройте forecast-backtest на последних 12–36 месяцах, а не только in-sample curve.

### Шаг 2. Добавить двухфазную часть и ML-остаток

Прогнозируйте liquid rate CRM, затем либо прикрепите fractional-flow/water-cut модуль, либо учите ML-коррекцию $(q_o,q_w)$ с $q^{CRM}_L$ и dynamics признаками. Сеть должна быть небольшой: masked GRU/TCN по producer или graph-GRU на sparse CRM graph. Параметры CRM остаются fixed в первом варианте; в более продвинутом варианте оптимизируются совместно через физически-информированный loss.

Контрольные сценарии в 48 траекториях следует разложить на train/validation/test **по траекториям**, например 32/8/8, а затем делать time-forward forecast внутри каждой. Все normalizers рассчитываются только на train. Для рассмотрения новых режимов отложите тестовые сценарии с самыми сильными изменениями закачки: это честнее, чем случайно делить месяцы.

### Шаг 3. Выбирать пространственный proxy только под точную задачу

Если цель — тысячи быстрых вариантов перераспределения закачки по известной модели, POD-ROM/POD-DEIM или CRM-hybrid рациональнее FNO. Если цель — быстрые предсказания только well QoI в переменном составе скважин, приоритет — GNN–LSTM с CRM-informed graph. Если цель — $p,S_w$ на сетке при вариации геологии, FNO/DeepONet становятся оправданными, но до запуска надо увеличить библиотеку: active learning, low-fidelity coarse-grid runs + ограниченное число fine-grid runs, а также transfer learning с близкого объекта.

> **Итоговое решение.** Для текущего масштаба рекомендуется поставить CRM как диагностический и оптимизационный baseline, затем сравнить его с CRM-residual masked GNN/GRU. Полный simulator proxy уровня FNO/DeepONet — это отдельный этап после увеличения разнообразия сценариев; 48 сценариев являются хорошим validation/seed набором, но не достаточным самостоятельным корпусом для универсального neural operator.

---

## 6. Таблица источников и применимость к задаче

В графе «объект» указаны только значения, найденные в открытом тексте/аннотации. Формулировка **«не раскрыто»** означает, что публикация/доступная аннотация не сообщает число скважин либо историю; она не заменяется догадкой.

| № | Источник, год | Объект: скважины и история/данные | Вывод в одну строку для 103 скв., 370 мес., 48 сценариев |
|---:|---|---|---|
| 1 | Yousef et al., 2006 [1] | Вертикальные скважины; открытая карточка не раскрывает единый размер кейса/горизонт. | **Да, как фундамент CRMIP/связности;** масштабировать только с sparsity и constraints. |
| 2 | Liang et al., 2007 [2] | 3D simulator; число скважин и история не раскрыты в abstract. | **Да,** как основа producer-based CRM и constrained allocation optimization. |
| 3 | Sayarpour et al., 2009 [3] | Синтетические и реальные examples; три control volumes, точные $N$/история не раскрыты в доступном abstract. | **Да,** главный источник для CRMT/CRMP/CRMIP и постановки под 370 месяцев. |
| 4 | Nguyen et al., 2011 [4] | Реальный Permian waterflood: **99 скважин**; после оптимизации — 1 год фактической проверки; общая история не раскрыта. | **Да, очень релевантно:** 99 почти равно 103, но нужны event-aware windows и QC. |
| 5 | Holanda et al., 2015 [5] | Сравнение ICRM, CRMP, CRMIP; размер кейсов в открытой карточке не раскрыт. | **Да,** полезно для выбора representation и sensitivity analysis. |
| 6 | Holanda et al., 2018 [6] | Обзор CRM, а не единый объект. | **Да, обязательный reference** для допущений, интерпретации $f$ и $\sum_j f_{ij}\le1$. |
| 7 | Sen, 2021 [7] | Два synthetic cases: **5 injectors + 4 producers**; срок истории не раскрыт. | **Частично:** полезно для CRM–RNN и constraints, но размер намного меньше целевого. |
| 8 | Maniglio et al., 2021 [8] | Малый synthetic + реальный зрелый объект; число скважин/история не раскрыты в abstract. | **Да, как hybrid/PINN template;** особенно при 48 сценариях и сильной регуляризации. |
| 9 | Characterizing Reservoir Connectivity…, 2016 [9] | Две synthetic модели с streaks/barriers; $N$/история не раскрыты; EUR error ≤3%. | **Частично:** подтверждает workflow «ANN/CRM connectivity → forecast», не масштабный benchmark. |
| 10 | van Doren et al., 2006 [10] | 2D heterogeneous waterflood: **2 025 ячеек, 4 050 states**, multiple wells; $N$/история не раскрыты. | **Да для POD-ROM,** если доступны snapshots из симулятора; 48 траекторий могут быть стартовой базой. |
| 11 | Yang et al., 2016 [11] | Two-phase 5-spot waterflood; в связанной карточке — 1 000-day snapshots; точное число runs не раскрыто. | **Да для POD-DEIM,** когда важны консервативность и full-physics states. |
| 12 | Tan et al., 2019 [12] | Две 2D heterogeneous water–oil модели; $N$/история не раскрыты в abstract. | **Да,** как reference adaptive snapshot/TDEIM, но не как прямой data-volume benchmark. |
| 13 | Jin et al., 2020 [13] | 2D heterogeneous oil–water, **9 скважин, 300 high-fidelity сценариев**. | **Погранично:** 48 < 300; возможен компактный physics-aware residual, но не аналог полной модели. |
| 14 | Coutinho et al., 2021 [14] | Oil–water, **4 injectors + 5 producers, 300 control sets** + validation. | **Погранично:** целевой граф больше, сценариев меньше; useful architecture, не готовый масштаб. |
| 15 | Huang et al., 2023 [15] | Oil–gas–water case: **189 скважин (96P+93I), 500 sims / 450 train**. | **Размер скважин релевантен, но 48 << 450;** CRM-informed sparse GNN — только pilot. |
| 16 | Du et al., 2022 [16] | Пять пластов, включая Bohai A Block B; $N$/история/scenarios не раскрыты в summary; MAPE 2.115%. | **Да для идеи динамической graph correction;** требуется directed weighted graph и своя валидация. |
| 17 | Ma et al., 2024 [17] | 2D: **1I+1P**, 241×241, 720 days, **1 000 realizations (800 train), 48 snapshots**. | **Нет для FNO с нуля:** 48 сценариев не равны 48 snapshots и заметно меньше 800 train realizations. |
| 18 | Diab & Al Kobaisi, 2024 [18] | CO$_2$–water; 96×200 spatial grid, 24 times, 12 channels; число скважин/realizations не раскрыто в доступном фрагменте. | **Частично:** полезна DeepONet architecture, но нужен simulator corpus/transfer для нефтяного waterflood. |
| 19 | Jiang & Durlofsky, 2023 [19] | Multi-fidelity flow surrogate; конкретный $N$/история зависят от case и не раскрыты в доступной карточке. | **Да, как стратегия:** coarse-grid массово + немного fine-grid — более реалистично, чем FNO на 48. |
| 20 | Jansen & Durlofsky, 2017 [20] | Методический обзор ROM для well-control optimization; не единый объект. | **Да,** для выбора между POD/TPWL и data-driven proxy и постановки speedup–accuracy trade-off. |

---

## References

[1]: https://onepetro.org/REE/article-abstract/9/06/630/196295 "Yousef et al. (2006), A Capacitance Model to Infer Interwell Connectivity from Production and Injection-Rate Fluctuations"
[2]: https://onepetro.org/SPEHEES/proceedings-abstract/07HEES/07HEES/SPE-107713-MS/142212 "Liang et al. (2007), Optimization of Oil Production Based on a Capacitance Model of Production and Injection Rates"
[3]: https://www.sciencedirect.com/science/article/pii/S0920410509002046 "Sayarpour et al. (2009), The Use of Capacitance–Resistance Models for Rapid Estimation of Waterflood Performance and Optimization"
[4]: https://doi.org/10.2118/146984-MS "Nguyen et al. (2011), Capacitance Resistive Model Application to Optimize Waterflood in a West Texas Field"
[5]: https://onepetro.org/SPELACP/proceedings-abstract/15LACP/15LACP/184134 "Holanda et al. (2015), Improved Waterflood Analysis Using the Capacitance-Resistance Model Within a Control Systems Framework"
[6]: https://doi.org/10.3390/en11123368 "Holanda et al. (2018), A State-of-the-Art Literature Review on Capacitance Resistance Models for Reservoir Characterization and Performance Forecasting"
[7]: https://arxiv.org/abs/2109.08779 "Sen (2021), Capacitance Resistance Model and Recurrent Neural Network for Well Connectivity Estimation — A Comparison Study"
[8]: https://doi.org/10.2118/207800-MS "Maniglio et al. (2021), Physics Informed Neural Networks Based on a Capacitance Resistance Model for Reservoirs Under Water Flooding Conditions"
[9]: https://onepetro.org/SPEWRM/proceedings-abstract/16WRM/16WRM/188267 "Characterizing Reservoir Connectivity and Forecasting Waterflood Performance Using Data-Driven and Reduced-Physics Models (2016)"
[10]: https://doi.org/10.1007/s10596-005-9014-2 "van Doren, Markovinović & Jansen (2006), Reduced-Order Optimal Control of Water Flooding Using Proper Orthogonal Decomposition"
[11]: https://onepetro.org/SJ/article-abstract/21/06/2141/206564 "Yang et al. (2016), Fast Multiscale Reservoir Simulations With POD-DEIM Model Reduction"
[12]: https://doi.org/10.1007/s10596-018-9782-0 "Tan et al. (2019), Trajectory-Based DEIM Model Reduction Applied to Reservoir Simulation"
[13]: https://doi.org/10.1016/j.petrol.2020.107273 "Jin, Liu & Durlofsky (2020), Deep-Learning-Based Surrogate Model for Reservoir Simulation With Time-Varying Well Controls"
[14]: https://doi.org/10.3389/fams.2021.651178 "Coutinho, Dall’Aqua & Gildin (2021), Physics-Aware Deep-Learning-Based Proxy Reservoir Simulation Model Equipped With State and Well Output Prediction"
[15]: https://doi.org/10.2118/215842-PA "Huang, Gong & Sun (2023), A Deep-Learning-Based GNN–LSTM Model for Reservoir Simulation and Optimization With Varying Well Controls"
[16]: https://doi.org/10.2118/210575-PA "Du et al. (2022), A Deep Learning Framework Using GCN for Adaptive Correction of Interwell Connectivity and GRU for Performance Prediction"
[17]: https://doi.org/10.1016/j.heliyon.2024.e38103 "Ma et al. (2024), Enhancing Subsurface Multiphase Flow Simulation With Fourier Neural Operator"
[18]: https://doi.org/10.1038/s41598-024-72393-0 "Diab & Al Kobaisi (2024), U-DeepONet: U-Net Enhanced Deep Operator Network for Geologic Carbon Sequestration"
[19]: https://doi.org/10.1016/j.jcp.2022.111800 "Jiang & Durlofsky (2023), Use of Multifidelity Training Data and Transfer Learning for Efficient Construction of Subsurface Flow Surrogate Models"
[20]: https://d-nb.info/1084056763/34 "Jansen & Durlofsky (2017), Use of Reduced-Order Models in Well Control Optimization"
