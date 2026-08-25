# Exp — пирамидальный механизм экспоненциального пространства

Комбинация `AoP (Addressing of Positions, src/vm1)` + `AoV (Addressing of Values, src/assoc)` в `exp`-пространство.

## Идея (README.md:19, §19)

`S0 = n·2ⁿ` бит — AoP-значения (позиции `vm1`). Каждый `V (|V|=256б)` сжимается `AoV` в `K (|K|=32б)`:

```
S0 --V->K--> S1  |S1| = |S0|·|K|/|V|
S1 --V->K--> S2
...
St = 1 ключ
t = ceil(log_{|V|/|K|}(n·2ⁿ/|K|))
```

Адрес — ключ любого уровня пирамиды. Восстановление — обратный проход `K -> окрестность -> ... -> S0` (цепочка AoV декодирований + AoP сборка).

`AoP` собирает `n·2ⁿ` бит итерациями по цепочке позиционных значений, `AoV` находит окрестность пачкой ключей.

## Состав

* `config.py` — `n, |V|, |K|, t`
* `pyramid.py` — `encode_pyramid(S0) -> [S0,S1,...,St]`, `decode_pyramid(key, levels) -> S0'`
* `tests/test_exp.py` — проверка `ham(S0, S0')` и `same-code`

Зависимости: `src/assoc/transformer/assoc_memory.py` (AoV), `src/vm1` (AoP, пока эмулируется списком позиций).

Статус: прототип pure-python, `n=8 → S0=2048б`, `V=256б → t=3`.
