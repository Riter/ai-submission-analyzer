# ai-submission-analyzer

Система онлайн-школы: приём сдач лабораторных, валидация и анализ (сходство, обратная связь по рубрике).

## Отчёты

| ЛР | Тема | Отчёт |
|----|------|-------|
| 1 | Приём работ и расширяемые анализаторы | [docs/reports/lab-1.md](docs/reports/lab-1.md) |

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

# анализ: импорт + все анализаторы из [analyzers].enabled → AnalysisReport на каждую пару (сдача, анализатор)
submission-analyzer analyze datasets/lab1/submissions.jsonl \
    --assignments datasets/lab1/assignments.json \
    --config configs/default.toml
```

### Анализаторы

| id | Что делает |
|----|-----------|
| `length` | число непустых строк текста и кода; пояснение в `rubric_feedback`, `score=null` |
| `template` | точные строки разрешённого шаблона в коде → `similarity` + evidence; совпадение с шаблоном разрешено |
| `requirements` | локальная проверка требований задания по маркерам (имена функций, литералы) |
| `model_stub` | заглушка модели: `InputPreparer → ModelClient → FeedbackFormatter`, тот же контракт отчёта |

Набор анализаторов задаётся в конфигурации, каждый отчёт содержит `analyzer_id`, `analyzer_version`
и `config_hash`. Формат отчёта — в [`docs/data_contract.md`](docs/data_contract.md).

### Как добавить свой анализатор

Наследоваться ни от чего не нужно — достаточно соблюсти протокол `Analyzer`
и зарегистрировать фабрику; `AnalysisService` при этом не меняется:

```python
from submission_analyzer.analyzers import build_report
from submission_analyzer.analysis.registry import default_registry

class MyAnalyzer:
    analyzer_id = "my"
    analyzer_version = "1.0.0"

    def analyze(self, submission, context):
        return build_report(self, submission, context, rubric_feedback=())

registry = default_registry()
registry.register("my", lambda config: MyAnalyzer())
# и добавить "my" в [analyzers].enabled
```

Вместо `submission-analyzer` можно использовать `python -m submission_analyzer`.
Код возврата: `0` — импорт выполнен (даже если часть строк отклонена), `2` — не читается файл или неверная конфигурация.

Валидацию можно вызвать и напрямую из Python, без CLI:

```python
from submission_analyzer.ingestion.service import IngestionService
from submission_analyzer.storage.repository import InMemorySubmissionRepository

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
  cli.py, __main__.py   CLI (тонкий адаптер: читает файлы, печатает JSON)
  core/                 доменные типы и общие утилиты — ни от чего в пакете не зависят
    models.py             Submission, AIDeclaration, Assignment (frozen dataclasses)
    errors.py             FormatError / SchemaError / ConflictError с деталями code, field, message
    reports.py            AnalysisReport, RubricFeedback, Evidence
    config.py             ProcessingConfig, режим ml | llm, набор анализаторов, config_hash
    hashing.py            content_hash, file_sha256
    textutils.py          source_lines, count_nonempty_lines
  ingestion/            приём сдач
    validation.py         parse_record, validate_record
    service.py            IngestionService: submit, import_numbered → ImportReport
    manifest.py           manifest корпуса
  storage/
    repository.py         SubmissionRepository (Protocol) + in-memory реализация
  adapters/             работа с файлами
    sources.py            JsonlFileSource / InMemorySource (контекстные менеджеры)
    loaders.py            чтение assignments.json
  analyzers/            протокол Analyzer + length, template, requirements, model_stub
  analysis/             запуск анализаторов
    registry.py           реестр фабрик анализаторов
    service.py            AnalysisService: запуск по реестру, изоляция ошибок
    baseline.py           детерминированный baseline для обоих режимов
```

Зависимости направлены к `core`, циклов нет: `storage`, `adapters`, `analyzers` → `core`;
`ingestion` → `core`, `storage`; `analysis` → `analyzers`, `core`; `cli` собирает всё вместе.
Файлы, stdin и сеть трогают только `adapters`, `cli` и `core.config.load_config`; доменные пакеты
(`core`, `ingestion`, `storage`, `analyzers`, `analysis`) работают с уже прочитанными данными.

## Документы
- [`docs/data_contract.md`](docs/data_contract.md) — формат данных, коды ошибок, manifest, AnalysisReport
- [`AI_USAGE.md`](AI_USAGE.md) — использование ИИ при разработке
