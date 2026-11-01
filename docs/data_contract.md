# Контракт данных

Преподаватель не выдал входные файлы и памятку, поэтому контракт зафиксирован здесь.
Код — источник истины: `src/submission_analyzer/core/models.py`, `ingestion/validation.py`, `core/errors.py`.

## Submission (одна строка JSONL = одна версия сдачи)

| Поле | Тип | Правила |
|------|-----|---------|
| `submission_id` | string | непустой; новая версия работы → новый id |
| `assignment_id` | string | непустой; если передан `assignments.json` — должен существовать |
| `student_id` | string | непустой |
| `submitted_at` | string | ISO-8601 **с часовым поясом** (`+03:00` или `Z`) |
| `text` | string | отчёт студента |
| `code` | string | исходный код (никогда не исполняется) |
| `ai_declaration` | object | `{used: bool, tools: [string], description: string}` |

- Другие поля (в том числе внутри `ai_declaration`) запрещены → `unknown_field`.
- `text` и `code` по отдельности могут быть пустыми, но не оба сразу.
- `used=false` при непустом `tools` → `invalid_ai_declaration`.
  `used=true` с пустым `description` при приёме **допустим**: достаточность декларации проверяет анализатор (этап Б).
- `ai_declaration` — самодекларация студента, **не метка происхождения текста**; её нельзя использовать
  как обучающий label.

### content_hash

`sha256:` + SHA-256 канонического JSON (сортированные ключи, без пробелов) всех полей; `submitted_at`
приведён к UTC. Один и тот же момент, записанный с разными смещениями, даёт один хеш.

### Приём (идемпотентность)

| Ситуация | Результат |
|----------|-----------|
| новый `submission_id` | `accepted` |
| тот же id, тот же `content_hash` | `duplicate` — возвращается уже принятая сдача, число сдач не растёт |
| тот же id, другой `content_hash` | ошибка `conflict / id_conflict`, сохранённая сдача не меняется |

## Ошибки

Каждая ошибка: `code`, `field`, `message`. В отчёте импорта добавляются `line_no`, `kind`, `submission_id`.

| kind (исключение) | code | Когда |
|-------------------|------|-------|
| `format` (`FormatError`) | `invalid_json` | строка — не JSON |
| | `not_an_object` | JSON, но не объект |
| `schema` (`SchemaError`) | `missing_field` | нет обязательного поля |
| | `unknown_field` | лишнее поле |
| | `invalid_type` | неверный JSON-тип |
| | `empty_value` | пустой id / нет ни text, ни code / пустой элемент tools |
| | `invalid_timestamp` | не ISO-8601 |
| | `naive_timestamp` | без часового пояса |
| | `invalid_ai_declaration` | противоречивая декларация |
| | `unknown_assignment` | задания нет в `assignments.json` |
| `conflict` (`ConflictError`) | `id_conflict` | id занят другим содержимым |

`SchemaError` содержит **все** ошибки записи, а не только первую.

## assignments.json

```json
{"assignments": [{"assignment_id": "...", "title": "...", "requirements": ["..."], "allowed_template": "..."}]}
```

## Конфигурация обработки (`configs/*.toml`)

```toml
[processing]
mode = "ml"                        # "ml" | "llm" (с ЛР 2 проект идёт по ML-ветке, llm — заглушка)
processing_version = "1.0.0"
prompt_template_version = "stub-0" # обязателен в llm-режиме
feature_schema_version = "pair-1"  # обязателен в ml-режиме: схема признаков пары

[analyzers]                        # необязательно; по умолчанию все встроенные
enabled = ["length", "template", "requirements", "token_similarity", "model_stub"]

[similarity]                       # необязательно (ЛР 2)
ngram = 3                          # длина токенной n-граммы
normalize_identifiers = true       # пользовательские имена -> ID, строковые литералы -> STR
top_k = 5                          # число кандидатов по умолчанию
methods = ["jaccard", "tfidf"]     # разрешённые методы сравнения
```

Неизвестные таблицы и ключи отклоняются (`ConfigError`), неизвестный id анализатора — при создании сервиса,
до обработки первой сдачи.

`config_hash` = `sha256:` от канонического JSON `{mode, processing_version, prompt_template_version,
feature_schema_version, analyzers, similarity}`. Одинаковый `config_hash` означает, что отчёты получены одной
конфигурацией и сравнимы.

На ЛР 1 оба режима возвращают детерминированный baseline: число непустых строк `text` и `code`,
модель не вызывается.

## Manifest корпуса

Создаётся `submission-analyzer import FILE --manifest PATH`. Содержит только идентификаторы и хеши,
без исходного текста/кода и без меток. Время запуска в manifest не пишется, поэтому повторный
импорт даёт байт-в-байт тот же файл.

```json
{
  "manifest_schema_version": "1",
  "processing": {"mode": "llm", "processing_version": "1.0.0", "prompt_template_version": "stub-0", "feature_schema_version": "stub-0"},
  "source": {"file": "submissions.jsonl", "sha256": "sha256:..."},
  "raw_content_fields": ["text", "code"],
  "feature_sets": [],
  "counts": {"accepted": 12, "duplicate": 1, "rejected_lines": 1},
  "entries": [{"line_no": 1, "status": "accepted", "submission_id": "sub-0001", "content_hash": "sha256:...", "error_codes": []}]
}
```

`feature_sets` пуст: признаки (ЛР 2) хранятся отдельно от исходных `text`/`code` с ключом `content_hash`.

## AnalysisReport

Общий контракт результата всех анализаторов (`src/submission_analyzer/core/reports.py`):

```json
{
  "submission_id": "sub-0002",
  "content_hash": "sha256:...",
  "analyzer_id": "template",
  "analyzer_version": "1.0.0",
  "config_hash": "sha256:...",
  "status": "succeeded",
  "score": null,
  "similarity": 0.25,
  "rubric_feedback": [{"criterion_id": "allowed_template", "status": null, "explanation": "...", "evidence": []}],
  "evidence": [{"source": "code", "start_line": 1, "end_line": 2, "note": "exact match with the allowed assignment template (permitted, not a violation)"}],
  "processing_error": null
}
```

- `status`: `succeeded` | `failed`. При исключении в анализаторе или невалидном отчёте сервис записывает
  `failed` с `processing_error`; остальные анализаторы продолжают работу.
- `score`, `similarity`: число в `[0, 1]` или `null` (недостаточно данных / не применимо).
- `rubric_feedback[].status`: `satisfied` | `not_satisfied` | `insufficient_evidence` | `null` (информационная заметка).
- `evidence`: диапазон строк **исходного** текста или кода, 1-based, включительно; строки считаются по `\n`
  как в редакторе. Сервис отклоняет отчёт, если диапазон выходит за пределы содержимого.
- `analyzer_version` у `model_stub` включает режим и версию промпта/схемы признаков, например `0.1.0+llm.stub-0`.
- Отчёт — это сигнал для преподавателя, а не вердикт о нарушении.

## Корпус ЛР 2 (`datasets/lab2/`)

Генерируется `scripts/generate_lab2_data.py` (seed 42). Сдачи — в том же формате `Submission`;
происхождение хранится **отдельно** и в анализаторы не передаётся: это описание того, как сделан учебный
пример, а не вердикт о плагиате.

`provenance.jsonl` — одна строка на сдачу:

```json
{"submission_id": "sub-0002", "family_id": "lab-01-fizzbuzz/f1", "origin": "reformatted", "derived_from": "sub-0001"}
```

- `family_id` — семейство: оригинальное решение и все его производные.
- `origin`: `original` | `exact_copy` | `reformatted` (отступы, комментарии, пустые строки) |
  `renamed` (переименование идентификаторов) | `resubmission` (повторная версия того же студента).

`pairs.jsonl` — размеченные пары разных студентов одного задания:

```json
{"a": "sub-0001", "b": "sub-0005", "assignment_id": "lab-01-fizzbuzz", "family_a": "lab-01-fizzbuzz/f1",
 "family_b": "lab-01-fizzbuzz/f2", "derivation_relation": "independent", "template_only": true, "split": "tune"}
```

- `derivation_relation`: `exact_copy` | `reformatted` | `renamed` (внутри семейства; для пары двух производных
  берётся более сильное преобразование) | `independent` (разные семейства).
- `template_only`: у независимой пары общие непустые строки есть, и все они — строки разрешённого шаблона.
- `split`: `tune` | `eval`. Пары между семействами разных сторон не размечаются.

`split.json`: `{"seed": 42, "families": {family_id: side}, "submissions": {submission_id: side}}`.
Семейство целиком лежит на одной стороне; `check_no_family_leakage` проверяет это для сдач и пар.

## Признаки и сравнение пар (ЛР 2)

- Признаки строятся из `tokenize`: отбрасываются `COMMENT`, `NL`, `NEWLINE`, `INDENT`, `DEDENT`, `ENCODING`,
  `ENDMARKER`. При `normalize_identifiers` пользовательские имена заменяются на `ID`, строки — на `STR`;
  ключевые слова, builtins и имена атрибутов после `.` сохраняются.
- Строка кода считается строкой шаблона, если её токены совпадают с токенами строки `allowed_template`
  (отступы и комментарии не мешают). N-граммы только из шаблонных токенов идут в `template_grams` и в score
  не входят.
- Код **не исполняется**: только `tokenize` и `ast.parse` (для флага `syntax_error`).
- Jaccard = |A ∩ B| / |A ∪ B| по множествам своих n-грамм; пустое объединение → `score = null`,
  `status = insufficient_data`.
- TF-IDF + cosine: словарь и IDF — только по стороне `tune` (`fit`), остальные сдачи — только `transform`.
  N-граммы, которых нет в словаре настройки, веса не имеют.

`PairResult` (команда `submission-analyzer similarity`):

```json
{
  "a": "sub-0001", "b": "sub-0002", "assignment_id": "lab-01-fizzbuzz", "kind": "cross_student",
  "method_id": "jaccard", "method_version": "1.0.0",
  "score": 1.0, "status": "scored", "shared_grams": 44, "template_shared": 1.0,
  "explanation": "44 n-grams of the students' own code are shared. ... not a verdict about a violation.",
  "evidence_a": [{"source": "code", "start_line": 2, "end_line": 12, "note": "n-grams shared with the other submission (outside the allowed template)"}],
  "evidence_b": [...],
  "warnings": []
}
```

- `kind`: `cross_student` (разные студенты) | `same_student` (версии одного студента — отдельный список).
  Пара сдачи с самой собой и пары разных заданий не строятся.
- `template_shared`: Jaccard по шаблонным n-граммам или `null`. Если общего нет ничего, кроме шаблона,
  `explanation` прямо говорит, что использование шаблона разрешено.
- `evidence_a` / `evidence_b`: диапазоны строк каждой сдачи, где лежат общие n-граммы (вне шаблона).
- `warnings`: например, сдача не разбирается `ast.parse` и сравнена только по токенам.

Признаки пары для ML (`similarity/pair_features.py`, схема `pair-1`): `jaccard`, `tfidf_cosine`,
`template_shared`, `shared_grams`, `size_ratio`, `token_ratio`, `has_parse_error`, `has_syntax_error`.
