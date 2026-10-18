# Контракт данных

Преподаватель не выдал входные файлы и памятку, поэтому контракт зафиксирован здесь.
Код — источник истины: `src/submission_analyzer/models.py`, `validation.py`, `errors.py`.

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
mode = "llm"                       # "llm" | "ml"
processing_version = "1.0.0"
prompt_template_version = "stub-0" # обязателен в llm-режиме
feature_schema_version = "stub-0"  # обязателен в ml-режиме
```

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
