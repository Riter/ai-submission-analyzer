# ai-submission-analyzer

> Выполнил студент 102 группы Иванов Илья

Система онлайн-школы: приём сдач лабораторных, валидация и анализ (сходство, обратная связь по рубрике).

## Отчёты

| ЛР | Тема | Отчёт |
|----|------|-------|
| 1 | Приём работ и расширяемые анализаторы | [docs/reports/lab-1.md](docs/reports/lab-1.md) |
| 2 | Анализ сходства и оценка качества | [docs/reports/lab-2.md](docs/reports/lab-2.md) |

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

Корпус для анализа сходства (ЛР 2) генерируется так же детерминированно:

```bash
python scripts/generate_lab2_data.py      # → datasets/lab2/
```

| Файл | Содержимое |
|------|-----------|
| `datasets/lab2/assignments.json` | 4 задания с разрешённым шаблоном |
| `datasets/lab2/submissions.jsonl` | 48 сдач: оригиналы, копии, переформатированные, переименованные и изменённые (`modified`) варианты, шаблон, SyntaxError, повторные версии |
| `datasets/lab2/provenance.jsonl` | происхождение каждой сдачи: `family_id`, `origin`, `derived_from` |
| `datasets/lab2/pairs.jsonl` | 110 размеченных пар: `derivation_relation`, `template_only`, `split` |
| `datasets/lab2/split.json` | сторона `tune` / `eval` каждого семейства и сдачи |

Формат записей, коды ошибок, формат manifest и корпуса — в [`docs/data_contract.md`](docs/data_contract.md).

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

# сходство (ЛР 2): потоковое чтение -> признаки tokenize -> пары одного задания разных студентов
submission-analyzer similarity datasets/lab2/submissions.jsonl \
    --assignments datasets/lab2/assignments.json --config configs/default.toml \
    [--method jaccard|tfidf] [--split datasets/lab2/split.json] [--submission sub-0001 --top-k 5] \
    [--threshold 0.59]

# эксперимент (ЛР 2, этап Б): пороги и классификатор review_required обучаются на tune, оцениваются на eval
submission-analyzer experiment --corpus datasets/lab2 --config configs/default.toml \
    --mlflow mlruns --out out/experiment.json
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # просмотр run: params, метрики, артефакты, модель

python scripts/run_seed_sweep.py          # разброс метрик по seed корпуса (нестабильность на малом наборе)
python scripts/profile_similarity.py      # wall time и peak memory на 100/1000/5000 записей
```

`--method tfidf` требует `--split`: векторизатор обучается только на стороне `tune`. Без `--submission`
печатаются все пары (`pairs`) и отдельно версии одного студента (`same_student_pairs`); с `--submission` —
top-k кандидатов без самой работы и без сдач других заданий. `--threshold` (например, порог из эксперимента)
добавляет `review_required` и строит evidence только для отмеченных пар. Score — сигнал для ручного разбора,
а не вердикт.

### Защита: преподаватель добавляет новый шаблон

1. Дописать строки в `allowed_template` нужного задания в `datasets/lab2/assignments.json`.
2. `submission-analyzer similarity ... --submission <id>`: строки нового шаблона уходят из `evidence` и из score,
   растёт число шаблонных n-грамм, пояснение говорит, что шаблон разрешён.
3. `submission-analyzer experiment --corpus datasets/lab2 --mlflow mlruns`: в новом run другой
   `sha256:assignments.json` — по нему отчёты со старым и новым шаблоном различаются; сравнить метрики и
   `template_only_max` двух run в `mlflow ui`. Проверка — `tests/test_template_change.py`.

### Анализаторы

| id | Что делает |
|----|-----------|
| `length` | число непустых строк текста и кода; пояснение в `rubric_feedback`, `score=null` |
| `template` | точные строки разрешённого шаблона в коде → `similarity` + evidence; совпадение с шаблоном разрешено |
| `requirements` | локальная проверка требований задания по маркерам (имена функций, литералы) |
| `token_similarity` | наибольший Jaccard по токенным 3-граммам с работой другого студента того же задания; шаблон не учитывается |
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
Код возврата: `0` — команда выполнена (даже если часть строк отклонена), `2` — не читается файл, неверная
конфигурация или аргументы.

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
python scripts/measure_index_memory.py    # память индекса признаков (ЛР 2)
python scripts/run_seed_sweep.py          # стабильность метрик эксперимента
```

## Структура

```
src/submission_analyzer/
  cli.py, __main__.py   CLI (тонкий адаптер: читает файлы, печатает JSON)
  core/                 доменные типы и общие утилиты — ни от чего в пакете не зависят
    models.py             Submission, AIDeclaration, Assignment (frozen dataclasses)
    errors.py             FormatError / SchemaError / ConflictError с деталями code, field, message
    reports.py            AnalysisReport, RubricFeedback, Evidence
    config.py             ProcessingConfig, режим ml | llm, анализаторы, [similarity], [experiment], config_hash
    corpus.py             происхождение примеров, размеченные пары, split, проверка утечки семейств
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
    loaders.py            чтение assignments.json и метаданных корпуса (provenance, pairs, split)
    corpus.py             загрузка каталога корпуса в индекс признаков + хеши файлов
    tracking.py           MLflow: params, метрики, артефакты разбиения, модель (SQLite в mlruns/)
  analyzers/            протокол Analyzer + length, template, requirements, token_similarity, model_stub
  similarity/           сходство кода (ЛР 2)
    features.py           токены tokenize -> n-граммы, шаблонные n-граммы отдельно
    methods.py            SimilarityMethod: Jaccard, TF-IDF + cosine (fit только на tune)
    pairs.py              кандидаты, PairResult с evidence, top-k
    pair_features.py      признаки пары для ML-классификатора (схема pair-1)
    index.py, pipeline.py индекс признаков и ленивый конвейер read -> validate -> features
  evaluation/           оценка сигнала review_required (ЛР 2, этап Б)
    dataset.py            размеченные пары + признаки; методы обучаются только на tune
    metrics.py            матрица ошибок, precision / recall / F1, доля неопределённых
    threshold.py          пороговый baseline (порог с лучшим F1 на tune)
    classifier.py         логистическая регрессия на признаках pair-1, проверка ответа модели, fallback
    experiment.py         сравнение детекторов на одном split, список ошибок
  analysis/             запуск анализаторов
    registry.py           реестр фабрик анализаторов
    service.py            AnalysisService: запуск по реестру, изоляция ошибок
    baseline.py           детерминированный baseline для обоих режимов
```

Зависимости направлены к `core`, циклов нет: `storage` → `core`; `ingestion` → `core`, `storage`;
`similarity` → `core`, `ingestion`; `evaluation` → `core`, `similarity`; `analyzers` → `core`, `similarity`;
`analysis` → `analyzers`, `core`; `adapters` (файлы, MLflow) → доменные пакеты; `cli` собирает всё вместе.
Файлы, stdin и MLflow трогают только `adapters`, `cli` и `core.config.load_config`; доменные пакеты
(`core`, `ingestion`, `storage`, `similarity`, `evaluation`, `analyzers`, `analysis`) работают с уже прочитанными
данными.

## Документы
- [`docs/data_contract.md`](docs/data_contract.md) — формат данных, коды ошибок, manifest, AnalysisReport
- [`AI_USAGE.md`](AI_USAGE.md) — использование ИИ при разработке
