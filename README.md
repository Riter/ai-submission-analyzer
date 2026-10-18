# ai-submission-analyzer

Система онлайн-школы: приём сдач лабораторных, валидация и анализ (сходство, обратная связь по рубрике).

## Установка

Требуется Python ≥ 3.11 (разработка ведётся на 3.13). Все зависимости объявлены в `pyproject.toml`,

**venv:**

```bash
git clone git@github.com:Riter/ai-submission-analyzer.git
cd ai-submission-analyzer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip      # нужен pip ≥ 21.3 для установки по pyproject.toml
pip install -e ".[dev]"
```

**conda (альтернатива):**

```bash
conda create -n python-lab python=3.13
conda activate python-lab
pip install -e ".[dev]"
```


## Данные

Датасет синтетический и генерируется детерминированно (seed=42):

```bash
python scripts/generate_lab1_data.py      # → datasets/lab1/
```

| Файл | Содержимое |
|------|-----------|
| `datasets/lab1/assignments.json` | 3 задания с разрешённым шаблоном кода |
| `datasets/lab1/submissions.jsonl` | 12 корректных сдач + дубликат + конфликт идентификатора |
| `datasets/lab1/invalid_submissions.jsonl` | по строке на каждый тип ошибки + 1 корректная строка в конце |

Формат записей, коды ошибок и формат manifest — в [`docs/data_contract.md`](docs/data_contract.md).

## Запуск

```bash
# импорт: JSON-отчёт {accepted, duplicate, errors, counts} в stdout
submission-analyzer import datasets/lab1/submissions.jsonl \
    --assignments datasets/lab1/assignments.json \
    --config configs/default.toml \
    --manifest out/manifest.json

# импорт файла с ошибками: неверные строки попадают в errors с номером строки, импорт не прерывается
submission-analyzer import datasets/lab1/invalid_submissions.jsonl --assignments datasets/lab1/assignments.json

# детерминированный baseline для режима из конфигурации (ml | llm)
submission-analyzer baseline datasets/lab1/submissions.jsonl --config configs/default.toml
```

Вместо `submission-analyzer` можно использовать `python -m submission_analyzer`.
Код возврата: `0` — импорт выполнен (даже если часть строк отклонена), `2` — не читается файл или неверная конфигурация.

Валидацию можно вызвать и напрямую из Python, без CLI:

```python
from submission_analyzer.repository import InMemorySubmissionRepository
from submission_analyzer.service import IngestionService

service = IngestionService(InMemorySubmissionRepository())
outcome = service.submit({...})   # IngestOutcome или SchemaError / ConflictError
```

## Проверка

```bash
pytest -q
```

## Структура

```
src/submission_analyzer/
  models.py      Submission, AIDeclaration, Assignment (frozen dataclasses)
  errors.py      FormatError / SchemaError / ConflictError с деталями code, field, message
  validation.py  parse_record, validate_record
  hashing.py     content_hash, file_sha256
  repository.py  SubmissionRepository (Protocol) + in-memory реализация
  service.py     IngestionService: submit, import_lines → ImportReport
  config.py      ProcessingConfig, режим ml | llm
  baseline.py    детерминированный baseline для обоих режимов
  manifest.py    manifest корпуса
  loaders.py     чтение assignments.json (файловый адаптер)
  cli.py         CLI
```

Доменный слой (`models`, `validation`, `service`, `repository`) не читает файлы, stdin и не обращается к сети.

## Документы
- [`docs/reports/lab-1.md`](docs/reports/lab-1.md) — отчёт по ЛР 1
- [`AI_USAGE.md`](AI_USAGE.md) — использование ИИ при разработке
