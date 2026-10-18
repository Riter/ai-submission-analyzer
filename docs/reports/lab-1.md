# Отчёт по ЛР 1 — Приём работ и расширяемые анализаторы

**Траектория:** AI и MLOps · **Уровень:** основной (углублённый не выполнялся)

## Этап А — приём и валидация (тег `lab-1-stage-a`)

### Выполненные требования

| Требование | Где реализовано |
|------------|-----------------|
| Пакет в src-layout, устанавливается в чистое окружение | `pyproject.toml`, `src/submission_analyzer/` |
| Модель `Submission` (неизменяемая) | `models.py` |
| Разбор JSON, валидация обязательных полей, часового пояса, AI-декларации | `validation.py` |
| Ошибка содержит `code`, `field`, `message`; вход не мутируется | `errors.py`, `test_input_record_is_not_mutated` |
| Различаются ошибка формата, нарушение схемы, конфликт id | `FormatError` / `SchemaError` / `ConflictError` |
| Тот же id + тот же хеш → уже принятая сдача; другой хеш → conflict | `service.py: IngestionService.submit` |
| In-memory репозиторий за интерфейсом | `repository.py` |
| Публичные функции аннотированы; домен не читает stdin и не ходит в HTTP | весь пакет; файлы читают только `cli.py`, `loaders.py`, `config.load_config` |
| **AI:** импорт корпуса + manifest с id, хешами и версией обработки | `manifest.py`, `cli import --manifest` |
| **AI:** исходные `text`/`code` отделены от будущих признаков | `Submission` хранит только сырой контент; `feature_sets: []` в manifest |
| **AI:** `ai_declaration` не используется как метка | manifest не содержит декларацию; `test_manifest_has_no_raw_content_or_provenance_labels` |
| **AI:** конфигурация режима ML/LLM, оба дают детерминированный baseline | `config.py`, `baseline.py`, `configs/default.toml` |

### Приёмка этапа А

| Сценарий | Тест |
|----------|------|
| Корректная запись принимается | `test_valid_record_is_accepted`, `test_new_submission_is_accepted` |
| Неверный timestamp | `test_invalid_timestamp`, `test_timestamp_without_timezone_is_rejected` |
| Пустой `student_id` | `test_empty_student_id` |
| Неизвестное поле | `test_unknown_field` |
| Неверные типы | `test_wrong_top_level_types`, `test_ai_declaration_validation` |
| Дубликат не увеличивает число сдач | `test_duplicate_returns_stored_submission_and_does_not_grow` |
| Конфликт не перезаписывает запись | `test_conflict_does_not_overwrite_original` |
| Повреждённая JSON-строка не останавливает импорт | `test_import_continues_after_corrupted_line`, `test_import_invalid_dataset_reports_every_scenario` |
| Установка в чистое окружение | ручная проверка (ниже) |

### Команды проверки

```bash
pip install -e ".[dev]"
pytest -q                                   # 79 passed

python scripts/generate_lab1_data.py        # детерминированный датасет (seed=42)
submission-analyzer import datasets/lab1/submissions.jsonl \
    --assignments datasets/lab1/assignments.json --config configs/default.toml --manifest out/manifest.json
# counts: accepted 12, duplicate 1, rejected_lines 1 (id_conflict для sub-0002)

submission-analyzer import datasets/lab1/invalid_submissions.jsonl --assignments datasets/lab1/assignments.json
# counts: accepted 1, rejected_lines 11 — по строке на каждый код ошибки

# установка в чистое окружение
python -m venv /tmp/cleanenv && /tmp/cleanenv/bin/pip install .
cd /tmp && /tmp/cleanenv/bin/submission-analyzer --version   # submission-analyzer 0.1.0
```

Повторный импорт с той же конфигурацией даёт байт-в-байт одинаковый manifest
(`test_import_writes_reproducible_manifest`); повторная генерация датасета — одинаковые файлы.

### Где выполняется валидация и почему аннотации её не заменяют

Валидация — в `validation.validate_record`; её вызывает `IngestionService.submit`, а CLI лишь передаёт
строки файла. Аннотации типов не проверяются интерпретатором во время выполнения: JSON из файла может
содержать `"student_id": 42`, и без явной проверки такой объект попал бы в `Submission`.

### Известные ограничения

- Репозиторий in-memory: данные живут только в рамках одного запуска CLI (хранение — ЛР 3).
- Файл читается целиком построчно в рамках одного импорта; некорректная UTF-8 кодировка прерывает импорт
  с кодом 2, а не отмечается построчно.
- Данные синтетические и небольшие; выводы о реальных студентах не делаются.

## Этап Б — контракты и расширения

_Будет заполнено после этапа Б._
