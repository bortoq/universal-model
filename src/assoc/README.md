# Ассоциативная адресация — standalone (без согласования с vm1)

Статус: 2026-08-24, прототип pure-python, recall 1.0

Временная замена — `encode(V)->K` (`|K|<|V|`) и `decode(K)->окрестность`. Пирамида `S0=n·2ⁿ -> ... -> 1 ключ`, `t=log`, см. §17-19 README.

## Что сделано (v0.1)

- `spec.md` — интерфейс `V(256б/32Б)->K(32б/4Б)`, сжатие 8x, метрики `recall@k`, `same-code`
- `tools/gen_dataset_pure.py` — uniform 1000×256бит, `ham 1-3`
- `tools/gen_dataset_clustered.py` — clustered 64×16=1024, `ham 1-5` (модель VM1-дампов)
- `transformer/baseline.py` — хеш-базлайн: `recall@10=1.0` exact, `noisy=0.03` → мотивирует VQ
- `transformer/train_pure.py` — pure-python tied-VQ k-means (без torch/numpy, Python 3.14 без wheel)
  - uniform 256 кодов: `same-code recall 1.0` (vn→ тот же K что v), `centroid ham 68.3`
  - clustered 64 кода: `same-code 1.0`, `centroid ham 6.3`, `exact_centroid 0.89`
- `transformer/codebook*.pkl` — сохранённые центроиды (tied, переиспользуемый кодбук)

## Почему pure-python

`Python 3.14` в окружении без `numpy/torch` wheel → обучение на `torch` перенесено в `model.py` (MLP+VQ, `1024×32`), запуск требует `pip install torch numpy` на `python3.11/3.12`. Текущий pure-python доказывает интерфейс и метрику.

## Следующие шаги

1. `torch` обучение `model.py` → `noisy recall@10 >=0.9` на uniform (LSH/Transformer+VQ)
2. Датасет из `vm1` дампов `S0=n·2ⁿ` вместо синтетики
3. Пирамида `S0->S1->...->1 ключ` `t=ceil(log_{|V|/|K|}(...))`
4. Согласование `pos->value->key->pos'` — отдельно, в другом коде

Запуск:
```
python3 src/assoc/tools/gen_dataset_pure.py
python3 src/assoc/transformer/baseline.py
python3 src/assoc/tools/gen_dataset_clustered.py
python3 src/assoc/transformer/train_pure.py --data src/assoc/data/synth_clustered.bin --codebook 64 --iters 10
```
