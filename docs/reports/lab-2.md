# Отчёт по ЛР 2 — Анализ сходства и оценка качества

**Траектория:** AI и MLOps, подвариант **ML и MLOps** · **Уровень:** основной (углублённый не выполнялся)

## Этап А — поток и baseline (тег `lab-2-stage-a`)

### Выполненные требования

| Требование | Где реализовано |
|------------|-----------------|
| Корпус ≥ 30 сдач и ≥ 40 размеченных пар, `assignment_id`, `family_id`, происхождение, разные студенты | `scripts/generate_lab2_data.py` → `datasets/lab2/`: 40 сдач, 18 семейств, 74 пары |
| ≥ 5 пар с разрешённым шаблоном | 34 независимые пары с `template_only = true` |
| Семейства разделены между настройкой и оценкой без утечки | `split.json` (19 сдач `tune`, 21 `eval`), `core/corpus.check_no_family_leakage` |
| `derivation_relation` и `template_only` — происхождение примера, а не вердикт | `pairs.jsonl`; `provenance.jsonl` лежит отдельно от `Submission` и не попадает в анализаторы |
| Чтение JSONL итератором, без загрузки корпуса целиком | `similarity/pipeline.py`: генераторы `iter_accepted → iter_features → build_index`, `IngestionService.iter_numbered` |
| Признаки через `tokenize` без комментариев и служебных токенов | `similarity/features.py: tokenize_code` |
| Строки шаблона учитываются отдельно | `CodeFeatures.template_grams`: в score не входят |
| Множества токенных 3-грамм, Jaccard; пустое объединение → `score = null` | `similarity/methods.py: JaccardMethod` |
| Сравнение работ одного задания разных студентов, без self-match, версии студента отдельно | `similarity/pairs.py: candidate_pairs`, `PairKind.SAME_STUDENT` |
| Пары с основаниями: диапазоны исходных строк и сведения о шаблоне | `PairResult.evidence_a/evidence_b`, `template_shared`, `explanation` |
| Описание расхода памяти индекса | раздел «Память индекса» ниже, `scripts/measure_index_memory.py` |
| **AI:** второй метод TF-IDF + cosine; `fit` только на настройке, оценка только `transform` | `similarity/methods.py: TfidfCosineMethod`, CLI `--method tfidf --split` |
| **ML:** признаки пары | `similarity/pair_features.py`, схема `pair-1` (`processing.feature_schema_version`) |
| Сравнение не заменено вопросом к LLM | всё сравнение локальное и детерминированное, модель не вызывается |

Анализ сходства подключён и к контракту ЛР 1: анализатор `token_similarity` (`analyzers/token_similarity.py`)
регистрируется в реестре и выдаёт наибольший Jaccard с работой другого студента. `AnalysisService` при этом
не менялся.

### Приёмка этапа А

| Сценарий | Тест |
|----------|------|
| Точная копия получает максимальное непустое сходство | `test_exact_copy_gets_maximal_non_empty_score` (оба метода), `test_similarity_command_on_corpus` |
| Общий шаблон не превращается в необъяснённое обвинение | `test_template_only_pair_is_explained_not_scored`, `test_template_is_excluded_from_the_score`, `test_token_similarity_template_only_is_not_an_accusation` |
| Пустые признаки обработаны | `test_empty_features`, `test_empty_features_give_null_score` |
| Некорректный код обработан | `test_untokenizable_code_gives_parse_error_not_exception`, `test_syntax_error_is_flagged_but_tokens_are_kept`, `test_syntax_error_is_reported_in_warnings` |
| Top-k не содержит саму работу и сдачи другого задания | `test_top_k_excludes_itself_own_versions_and_other_assignments`, `test_similarity_top_k` |
| Evidence указывает на существующие строки | `test_evidence_points_to_existing_lines_of_each_side`, `test_gram_line_spans_point_to_source_lines` |

Обязательные проверки ЛР 2, которые относятся к этапу А:

| Проверка | Тест |
|----------|------|
| Одинаковый код | `test_exact_copy_gets_maximal_non_empty_score` |
| Переименованные переменные | `test_renamed_variables_stay_highly_similar`, `test_without_normalization_renaming_lowers_jaccard` |
| Независимые решения | `test_independent_solutions_score_lower_than_derived` |
| Разрешённый шаблон | `test_template_only_pair_is_explained_not_scored`, `test_commented_template_line_still_counts_as_template` |
| Пустой ввод | `test_empty_features`, `test_empty_features_give_null_score` |
| SyntaxError | `test_syntax_error_is_flagged_but_tokens_are_kept`, `test_broken_code_gives_null_score_with_reason` |
| Self-match | `test_no_self_match_and_no_cross_assignment_pairs` |
| Повторная версия студента | `test_versions_of_one_student_are_compared_separately` |
| Отсутствие утечки семейств | `test_no_family_leakage_between_tune_and_eval`, `test_leakage_check_detects_a_family_on_both_sides` |
| Ленивое чтение | `test_reading_is_lazy`, `test_nothing_is_read_before_iteration` |
| TF-IDF не переобучается на оценке | `test_tfidf_scoring_does_not_change_vocabulary_or_idf`, `test_tfidf_cannot_be_refitted` |
| Код студента не исполняется | `test_code_is_never_executed` |

Проверки невалидного результата модели и воспроизводимости эксперимента относятся к этапу Б
(классификатор и MLflow).

### Как устроен корпус

На каждое из 4 заданий написано 4 независимых решения. Решение — это оригинал семейства; к нему программно
добавлены производные: точная копия, переформатирование (отступы в 2 пробела, комментарии, пустые строки) и
переименование идентификаторов через `tokenize`. Набор производных у семейств разный. Ещё есть две особые
сдачи: решение, почти целиком состоящее из шаблона, и решение с `SyntaxError`. Также есть две повторные
версии одного студента. Все сдачи используют шаблон задания, поэтому у большинства независимых пар общие
строки — только строки шаблона.

Семейства каждого задания перемешиваются с seed 42 и делятся пополам между `tune` и `eval`. Пара получает
сторону своих семейств; пары между семействами разных сторон не размечаются, иначе одна работа оказалась бы
по обе стороны.

### Память индекса

Индекс (`FeatureIndex`) хранит для каждой сдачи словарь «n-грамма → диапазоны строк» и множество шаблонных
n-грамм; исходный `text`/`code` в нём не хранится. Объём — O(N · G), где N — число сдач, G — число различных
n-грамм в сдаче (здесь около 70–80). Потоковое чтение экономит память только на входе: одновременно
обрабатывается одна строка файла. Сам индекс при этом растёт линейно с корпусом, поэтому константным по памяти
конвейер не становится.

Замер (`python scripts/measure_index_memory.py --sizes 40 1000 --repeats 3`, `tracemalloc`, медиана из 3
повторов, seed 42, ngram 3; Python 3.13.16, macOS 15.7 arm64). Для размера 1000 корпус дополнен
синтетическими копиями: они нужны только для нагрузки.

| Сдач | n-грамм всего | Индекс, KiB | Пик при построении, KiB | KiB на сдачу |
|------|---------------|-------------|--------------------------|--------------|
| 40 | 2 818 | 545.9 | 652.3 | 13.6 |
| 1000 | 77 531 | 15 046.8 | 15 786.2 | 15.0 |

Около 190 байт на n-грамму: кортеж из трёх строк, ключ словаря и кортеж диапазонов. Строки токенов
(`ID`, `(`, `for` …) интернированы и почти не занимают место. Если индекс станет узким местом, n-граммы
можно хешировать в целые числа, а диапазоны строк хранить только на время построения evidence.
Профилирование 100/1000/5000 записей выполняется на этапе Б.

### Команды проверки

```bash
pip install -e ".[dev]"                   # ставит scikit-learn
pytest -q                                 # 353 passed

python scripts/generate_lab2_data.py      # корпус datasets/lab2/ (seed=42), повторный запуск даёт те же файлы

# все пары одного задания, Jaccard
submission-analyzer similarity datasets/lab2/submissions.jsonl \
    --assignments datasets/lab2/assignments.json --config configs/default.toml

# TF-IDF: fit только на стороне tune из split.json
submission-analyzer similarity datasets/lab2/submissions.jsonl \
    --assignments datasets/lab2/assignments.json --config configs/default.toml \
    --method tfidf --split datasets/lab2/split.json

# top-k кандидатов для одной сдачи
submission-analyzer similarity datasets/lab2/submissions.jsonl \
    --assignments datasets/lab2/assignments.json --submission sub-0001 --top-k 5

python scripts/measure_index_memory.py    # память индекса
```

Контрольный прогон по всему корпусу. Это не оценка качества: порог не подбирался, разбиение на
`tune`/`eval` здесь не учитывается.

| Тип пары | Пар | Jaccard: мин – медиана – макс | TF-IDF: мин – медиана – макс |
|----------|-----|-------------------------------|------------------------------|
| exact_copy | 4 | 1.000 – 1.000 – 1.000 | 1.000 – 1.000 – 1.000 |
| reformatted | 12 | 1.000 – 1.000 – 1.000 | 1.000 – 1.000 – 1.000 |
| renamed | 12 | 1.000 – 1.000 – 1.000 | 1.000 – 1.000 – 1.000 |
| independent, template_only | 34 | 0.048 – 0.200 – 0.561 | 0.043 – 0.224 – 0.849 |
| independent, остальные | 12 | 0.138 – 0.178 – 0.561 | 0.146 – 0.194 – 0.849 |

### Известные ограничения

- **Нормализация имён склеивает переименование с копией.** При `normalize_identifiers = true` у
  переименованного варианта те же n-граммы, что у оригинала, поэтому score равен 1.0. С
  `normalize_identifiers = false` переименование снижает score (`test_without_normalization_renaming_lowers_jaccard`).
  Это настройка конфигурации, и она входит в `config_hash`.
- **Синтетические производные слишком простые.** Все три вида преобразований не меняют последовательность
  токенов, поэтому на этом корпусе они отделяются от независимых решений тривиально. Выводы о
  реальных студентах по такому корпусу делать нельзя.
- **Независимые решения на одной идиоме близки.** Самая похожая независимая пара — `sub-0010` и `sub-0038`
  (0.56 по Jaccard, 0.85 по TF-IDF). Обе считают слова циклом `counts[word] = counts.get(word, 0) + 1`.
  Сходство здесь — сигнал для ручного разбора, а не доказательство копирования. TF-IDF усиливает такие пары:
  общие n-граммы редки в наборе настройки и получают большой вес.
- **Jaccard чувствителен к длине.** Короткая работа, целиком содержащаяся в длинной, получает низкий score.
- **TF-IDF не видит n-граммы, которых нет в наборе настройки.** Если у пары нет ни одной n-граммы из словаря
  `tune`, результат — `insufficient_data`.
- **Сравнение — полный перебор пар внутри задания**, O(n²) на задание. Индекс держится в памяти целиком.
- **`ast.parse` используется только для флага `syntax_error`.** Сравнение структуры AST не реализовано: это
  вариант Python Core, а не AI.
