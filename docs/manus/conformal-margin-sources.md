Ключевые источники и выводы:

1. Ryan Tibshirani, Conformal Prediction, UC Berkeley lecture, 2023, https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf. Конформный вывод даёт finite-sample marginal coverage для нового i.i.d./exchangeable наблюдения; split conformal строится через порядковую статистику calibration scores.

2. Ruiting Liang, Wanrong Zhu, Rina Foygel Barber, Conformal prediction after data-dependent model selection, arXiv:2408.07066v3, 2025/2026 version, https://arxiv.org/html/2408.07066v3. Абстракт и введение прямо указывают, что использование одного hold-out набора для выбора модели и последующей калибровки приводит к потере покрытия из-за selection bias; отдельно обсуждаются дополнительные sample splitting и методы пост-селекционной валидности с finite-sample guarantees.

Использовать эти источники в отчёте вместе с первичными публикациями по weighted/local conformal и conditional coverage.

3. Jing Lei, Max G'Sell, Alessandro Rinaldo, Ryan J. Tibshirani, Larry Wasserman, Distribution-Free Predictive Inference for Regression, JASA 2018, https://www.stat.berkeley.edu/~ryantibs/papers/conformal-jasa.pdf. Статья формулирует marginal finite-sample coverage для нового exchangeable наблюдения и рассматривает residual-based и locally varying length intervals; важна оговорка, что это не условное покрытие по конкретному x без дополнительных условий.

4. Rohan Hore, Rina Foygel Barber, Conformal prediction with local weights: randomization enables robust guarantees, arXiv v2 2024, https://arxiv.org/html/2310.07850v2. Обычное CP даёт marginal coverage, усреднённое по тестовым covariates; локальное/условное покрытие существенно сложнее. RLCP получает relaxed local/test-conditional guarantees при i.i.d. и явных условиях на локализацию/границы областей, а не простым делением остатка на distance.

5. Ryan J. Tibshirani, Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, Conformal Prediction Under Covariate Shift, arXiv v3 2020 (paper first posted 2019), https://arxiv.org/html/1904.06019. Weighted CP extends validity beyond ordinary exchangeability only when the shift/likelihood ratio is known or accurately estimated, or when weighted exchangeability holds. Ordinary unweighted calibration is not automatically valid under changed test-selection distribution.
