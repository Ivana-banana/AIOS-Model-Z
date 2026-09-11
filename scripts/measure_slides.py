"""Замерить, помещается ли каждый слайд колоды в сцену 1920×1080.

    python scripts/measure_slides.py

Зачем отдельный скрипт. `docs/slides.html` проверяет себя сам — `checkOverflow`
рисует красную рамку «СЛАЙД НЕ ПОМЕСТИЛСЯ», — но только у **показанного** слайда
и только в браузере у человека. В печати рамка скрыта намеренно
(`@media print`), поэтому в PDF, который уходит жюри, лишнее просто **срезается
молча**: 02.09 так уже пропал пункт со слайда 5.

Отсюда правило Р-207: «слайды стоит перемерить на машине с браузером перед
защитой». Замер 05.09 показал, чего стоит это «стоит»: два слайда из двенадцати
не помещались — главный «Заявляемое число» на 44 пикселя и запасной 3 на 15.
Первый и есть тот слайд, ради которого проводится защита.

Как считает. Копия колоды кладётся во временный каталог, к ней дописывается
скрипт замера: он зовёт **собственную** функцию колоды `show(i)` по каждому
слайду и читает её же признак `overflowing`. То есть меряется не наша выдумка о
вёрстке, а ровно то, что увидит человек. Сама `docs/slides.html` не меняется.

Код возврата 1, если хоть один слайд не поместился, — чтобы шаг можно было
поставить в проверку перед защитой, а не читать глазами.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Где искать браузер. Chrome и Edge — оба Chromium, `--dump-dom` есть у обоих.
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

PROBE = """
<script>
addEventListener('load', () => setTimeout(() => {
  const all = Array.from(document.querySelectorAll('.deck-stage > section'));
  const out = [];
  for (let i = 0; i < all.length; i++) {
    show(i);
    const body = all[i].querySelector('.slide-body');
    out.push({
      n: i + 1,
      stamp: (all[i].querySelector('.stamp') || {}).textContent || '',
      flagged: all[i].classList.contains('overflowing'),
      over: body.scrollHeight - body.clientHeight,
    });
  }
  document.title = 'ЗАМЕР' + JSON.stringify(out);
}, 800));
</script>
"""


#: Кэш браузеров playwright: `playwright install chromium` кладёт Chromium сюда,
#: в профиль пользователя, а не в систему. Для машины без Chrome и Edge это
#: единственный способ получить браузер, не ставя его всем пользователям и не
#: требуя прав администратора — а замер обязателен перед защитой (Р-232).
#: Каталог версионирован (`chromium-1200`), поэтому ищется маской.
PLAYWRIGHT_CACHES = (
    Path.home() / "AppData" / "Local" / "ms-playwright",
    Path.home() / ".cache" / "ms-playwright",
    Path.home() / "Library" / "Caches" / "ms-playwright",
)
PLAYWRIGHT_BINARIES = (
    "chromium-*/chrome-win64/chrome.exe",
    "chromium-*/chrome-win/chrome.exe",
    "chromium-*/chrome-linux/chrome",
    "chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
)


#: Признак «браузера на этой машине считать, что нет».
#:
#: `PLAYWRIGHT_BROWSERS_PATH` прячет только кэш playwright и не может спрятать
#: Chrome, установленный в систему: его находит `shutil.which` первой же
#: строкой. Значит, клон с системным браузером снова отвечал бы 21 пропуском
#: против обещанных жюри 22 — ровно тем отказом, который Р-259 и закрывал,
#: только другим путём (Р-262). Спрятать браузер нечем, кроме явного признака:
#: подменять `PATH` нельзя — по нему подпроцесс ищет и сам Python.
NO_BROWSER = "AIOS_NO_BROWSER"


def browser() -> str | None:
    """Путь к Chromium-браузеру или None, если его на машине нет."""
    if os.environ.get(NO_BROWSER):
        return None
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    for path in BROWSERS:
        if Path(path).is_file():
            return path
    # `PLAYWRIGHT_BROWSERS_PATH` — переменная самого playwright, а не наша
    # выдумка: ею задают каталог браузеров. Её же выставляет `clean_clone_check`,
    # чтобы клон мерил ту среду, которую README обещает жюри, — установку по
    # `requirements-dev.txt`, где браузера нет.
    chosen = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    caches = (Path(chosen),) if chosen else PLAYWRIGHT_CACHES
    for cache in caches:
        for pattern in PLAYWRIGHT_BINARIES:
            # sorted — чтобы из нескольких установленных версий бралась одна и
            # та же: замер обязан повторяться, а не зависеть от порядка обхода
            for path in sorted(cache.glob(pattern)):
                if path.is_file():
                    return str(path)
    return None


def measure(deck: Path, chrome: str, timeout: int = 120) -> list[dict]:
    """Прогнать колоду в headless-браузере и вернуть замер по каждому слайду."""
    with tempfile.TemporaryDirectory() as room:
        copy = Path(room) / "measure.html"
        copy.write_text(deck.read_text(encoding="utf-8") + PROBE, encoding="utf-8")
        done = subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             "--window-size=1920,1080", "--virtual-time-budget=9000",
             "--dump-dom", copy.resolve().as_uri()],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    found = re.search(r"ЗАМЕР(\[.*?\])</title>", done.stdout, re.DOTALL)
    if not found:
        raise RuntimeError(
            "браузер не вернул замер: колода не выполнила свой скрипт "
            f"(код {done.returncode}). Первые строки вывода:\n{done.stdout[:300]}"
        )
    return json.loads(found.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Колода задаётся явно. Умолчанием была `docs/slides.html` — наша колода в
    # репозитории; 11.09 она удалена, действующая презентация собирается вне
    # репозитория. Умолчание на несуществующий файл молча роняло бы запуск без
    # аргумента, поэтому путь обязателен.
    parser.add_argument("deck", type=Path)
    args = parser.parse_args(argv)

    if not args.deck.is_file():
        print(f"колоды нет: {args.deck}", file=sys.stderr)
        return 2
    chrome = browser()
    if chrome is None:
        print("браузера на этой машине нет — мерить нечем; "
              "замер обязателен на машине с Chrome перед защитой", file=sys.stderr)
        return 2

    slides = measure(args.deck, chrome)
    over = [one for one in slides if one["flagged"]]
    for one in slides:
        mark = "НЕ ПОМЕСТИЛСЯ" if one["flagged"] else "помещается"
        print(f"  {one['n']:2d}. {one['stamp']:32s} {mark}"
              + (f", лишних {one['over']} px" if one["flagged"] else ""))
    print()
    if over:
        print(f"слайдов не поместилось: {len(over)} из {len(slides)}. "
              "В печати рамка скрыта, значит в PDF содержимое срежется молча.",
              file=sys.stderr)
        return 1
    print(f"все {len(slides)} слайдов помещаются в сцену 1920×1080")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
