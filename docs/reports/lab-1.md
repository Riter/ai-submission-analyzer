# Отчёт по ЛР 1 — Приём работ и расширяемые анализаторы

**Траектория:** AI и MLOps · **Уровень:** основной (углублённый не выполнялся)

## Этап А — приём и валидация (тег `lab-1-stage-a`)

### Выполненные требования

| Требование | Где реализовано |
|------------|-----------------|
| Пакет в src-layout, устанавливается в чистое окружение | `pyproject.toml`, `src/submission_analyzer/` |
| Модель `Submission` (неизменяемая) | `core/models.py` |
| Разбор JSON, валидация обязательных полей, часового пояса, AI-декларации | `ingestion/validation.py` |
| Ошибка содержит `code`, `field`, `message`; вход не мутируется | `core/errors.py`, `test_input_record_is_not_mutated` |
| Различаются ошибка формата, нарушение схемы, конфликт id | `FormatError` / `SchemaError` / `ConflictError` |
| Тот же id + тот же хеш → уже принятая сдача; другой хеш → conflict | `ingestion/service.py: IngestionService.submit` |
| In-memory репозиторий за интерфейсом | `storage/repository.py` |
| Публичные функции аннотированы; домен не читает stdin и не ходит в HTTP | весь пакет; файлы читают только `cli.py`, `adapters/`, `core/config.load_config` |
| **AI:** импорт корпуса + manifest с id, хешами и версией обработки | `ingestion/manifest.py`, `cli import --manifest` |
| **AI:** исходные `text`/`code` отделены от будущих признаков | `Submission` хранит только сырой контент; `feature_sets: []` в manifest |
| **AI:** `ai_declaration` не используется как метка | manifest не содержит декларацию; `test_manifest_has_no_raw_content_or_provenance_labels` |
| **AI:** конфигурация режима ML/LLM, оба дают детерминированный baseline | `core/config.py`, `analysis/baseline.py`, `configs/default.toml` |

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
pytest -q                                   # 79 passed на момент тега lab-1-stage-a

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

Валидация — в `ingestion/validation.validate_record`; её вызывает `IngestionService.submit`, а CLI лишь передаёт
строки файла. Аннотации типов не проверяются интерпретатором во время выполнения: JSON из файла может
содержать `"student_id": 42`, и без явной проверки такой объект попал бы в `Submission`.

### Известные ограничения

- Репозиторий in-memory: данные живут только в рамках одного запуска CLI (хранение — ЛР 3).
- Файл читается целиком построчно в рамках одного импорта; некорректная UTF-8 кодировка прерывает импорт
  с кодом 2, а не отмечается построчно.
- Данные синтетические и небольшие; выводы о реальных студентах не делаются.

## Этап Б — контракты и расширения (тег `lab-1`)

### Выполненные требования

| Требование | Где реализовано |
|------------|-----------------|
| Контракт `Analyzer.analyze(submission, context) -> AnalysisReport` (синхронный) | `analyzers/base.py` (`Protocol`) |
| Context передаёт требования `Assignment` и доступ к данным через интерфейс | `AnalysisContext(assignment, submissions: SubmissionReader, config, config_hash)` |
| В отчёте `analyzer_id`, `analyzer_version` | `core/reports.py: AnalysisReport`, `build_report` |
| `LengthAnalyzer`: непустые строки текста и кода, пояснение в `rubric_feedback`, `score=null` | `analyzers/length.py` |
| `TemplateAnalyzer`: точные строки шаблона → `similarity` + пояснение, что совпадение разрешено | `analyzers/template.py` |
| Номера строк evidence относятся к исходному содержимому | `core/textutils.source_lines`, `test_template_line_numbers_refer_to_original_content` |
| JSONL-источник и in-memory источник | `adapters/sources.py: JsonlFileSource, InMemorySource` |
| Один набор контрактных тестов на оба источника и на все анализаторы | `test_source_contract.py`, `test_analyzer_contract.py` |
| Центральный сервис выбирает анализатор по реестру, без ветвления по классам | `analysis/registry.py`, `analysis/service.py`, `test_service_has_no_branching_on_concrete_analyzers` |
| **AI:** интерфейсы подготовки входа и формирования feedback | `analyzers/model_stub.py: InputPreparer, FeedbackFormatter` (+ `ModelClient`) |
| **AI:** локальный анализатор требований | `analyzers/requirements.py` |
| **AI:** stub модели с тем же контрактом результата | `ModelStubAnalyzer` + `StubModelClient`; проходит тот же `test_analyzer_contract.py` |
| **AI:** `config_hash` | `core/config.config_hash`; есть в каждом отчёте и в manifest |
| **AI:** LLM — версия шаблона запроса, ML — версия схемы признаков | `analyzer_version` stub: `0.1.0+llm.<prompt>` / `0.1.0+ml.<features>` |

### Обязательные проверки ЛР 1

| Проверка | Тест |
|----------|------|
| Валидный импорт | `test_import_valid_dataset` |
| Повреждённый JSON | `test_corrupted_json_is_format_error`, `test_import_continues_after_corrupted_line` |
| Неверные типы | `test_wrong_top_level_types`, `test_ai_declaration_validation` |
| Дубликат | `test_duplicate_returns_stored_submission_and_does_not_grow` |
| Конфликт | `test_conflict_does_not_overwrite_original` |
| Ошибка анализатора | `test_analyzer_error_does_not_stop_other_analyzers`, `test_invalid_model_output_is_rejected` |
| Освобождение ресурса | `test_closed_on_exception`, `test_closed_on_early_exit`, `test_file_closed_when_import_raises` |
| Диапазоны evidence | `test_evidence_points_to_existing_source_lines`, `test_report_with_out_of_range_evidence_is_rejected` |
| Неизменность исходного объекта | `test_input_record_is_not_mutated`, `test_submission_is_not_modified` |

### Подготовка к защите

Третий анализатор (непустая AI-декларация при `used=true`) подключается через `registry.register(...)` и
`[analyzers].enabled` — **без изменения `analysis/service.py`**. Отрепетировано в
`tests/test_analysis_service.py::test_third_analyzer_plugs_in_via_registry`.

**Protocol vs наследование.** `Analyzer` — структурный тип: класс считается анализатором, если у него есть
`analyzer_id`, `analyzer_version` и `analyze` нужной сигнатуры. Наследоваться от базового класса не
требуется, поэтому анализатор из стороннего пакета или теста не зависит от нашей иерархии. При наследовании
подкласс получает реализацию родителя и связан с ним; здесь общего поведения нет, есть только контракт —
поэтому выбран `Protocol`, а общие поля отчёта заполняет функция `build_report` (композиция, а не базовый класс).

**Кто управляет ресурсом.** Файл открывает и закрывает `JsonlFileSource` в `__enter__`/`__exit__`; его
использует CLI в блоке `with`. `IngestionService` получает только итератор пар `(line_no, raw)` и о файле
не знает. `__exit__` закрывает файл и при исключении, и при `break`, исключение не подавляется.

**Какие исключения видит пользователь.**
- Через CLI — ни одного трейсбека. Ошибки записей попадают в JSON-отчёт (`errors[]` с `kind/code/field/message`).
  Ошибки конфигурации и файлов → сообщение в stderr и код возврата 2. Сбой анализатора → отчёт `failed`
  с `processing_error`.
- При вызове из Python — доменные `SchemaError` / `ConflictError` из `IngestionService.submit` и
  `ConfigError` при создании `AnalysisService` с неизвестным анализатором.

### Команды проверки

```bash
pytest -q                                   # 240 passed

submission-analyzer analyze datasets/lab1/submissions.jsonl \
    --assignments datasets/lab1/assignments.json --config configs/default.toml
# 12 принятых сдач × 4 анализатора = 48 отчётов, все succeeded, единый config_hash
```

Повторный запуск даёт идентичный вывод; при `mode = "ml"` меняются `config_hash` и `analyzer_version`
у `model_stub` (`test_analyze_is_reproducible_and_mode_changes_config_hash`). Установка `pip install .`
в чистый venv и запуск `analyze` из другого каталога проверены вручную.

### Измерения

Время и память по требованиям ЛР 1 не измеряются (профилирование — ЛР 2). Полный набор тестов
выполняется примерно за 0,2 с; анализ датасета (14 строк, 48 отчётов) занимает доли секунды.

### Известные ограничения (этап Б)

- `RequirementsAnalyzer` — простая эвристика по маркерам. Требования без имён функций и литералов
  («Report explains the approach») всегда получают `insufficient_evidence`. Наличие маркера не означает,
  что требование действительно выполнено.
- `TemplateAnalyzer` сравнивает строки целиком после `strip`. Общие короткие строки (`return result`)
  тоже засчитываются как совпадение с шаблоном.
- Анализ выполняется последовательно и синхронно (конкурентный адаптер — ЛР 3).
