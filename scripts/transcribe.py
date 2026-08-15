"""Расшифровка записей встреч организаторов в текст.

    python scripts/transcribe.py materials/Запись_встречи_10_08_2026.webm

Записи в репозиторий не кладутся (по 0,5–0,8 ГБ при пределе GitHub в 100 МБ),
поэтому в `docs/method/transcripts/` лежит их текст, а здесь — команда, которой
он получен. Самое ценное в этих записях — блок вопросов и ответов: там звучит
то, чего нет ни в презентациях, ни в методике (`docs/method/qa-organisers.md`).

Формат вывода повторяет уже сделанные расшифровки: две строки шапки с исходным
файлом и моделью, дальше реплики с отметкой времени.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "docs" / "method" / "transcripts"


def timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="файл записи (.webm, .mp4, .m4a …)")
    parser.add_argument("--model", default="base",
                        help="модель whisper: base хватает для речи с экрана (по умолчанию base)")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--out", default=None,
                        help="куда писать (по умолчанию docs/method/transcripts/<имя>.txt)")
    arguments = parser.parse_args(argv)

    source = Path(arguments.source)
    if not source.is_file():
        print(f"нет файла: {source}", file=sys.stderr)
        return 1

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("нужен faster-whisper: pip install -r requirements-dev.txt", file=sys.stderr)
        return 1

    target = Path(arguments.out) if arguments.out else TRANSCRIPTS / f"{source.stem}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(arguments.model, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(source), language=arguments.language)
    print(f"длительность: {timestamp(info.duration)}, модель {arguments.model}")

    # пишем по мере расшифровки: часовая запись идёт долго, и обрыв на середине
    # не должен стоить всей работы
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"-- {source.name}\n-- модель whisper {arguments.model}, "
                     f"язык {arguments.language}\n\n")
        for segment in segments:
            line = f"[{timestamp(segment.start)}] {segment.text.strip()}"
            handle.write(line + "\n")
            handle.flush()
            print(line)

    print(f"\nрасшифровка: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
