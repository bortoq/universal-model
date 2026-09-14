# Памятка: как корректно ставить эксперименты AoV

**Цель:** 1 вопрос = 1 эксперимент. Меняется только 1 переменная.

## 1. Фиксируем (не трогать между прогонами)
- Датасет: N_train, N_hold, V_BYTES, распределение (clustered/uniform), seed, FLIPS
- Обучение: epochs, batch, lr, optimizer, loss (BCE+ball), seed
- Оценка: N_eval, метрики (ham mean/median/p90, same_pt, same_ball, ball best)
- Железо: CPU/GPU, время

## 2. Варьируем (одно)
- Пример: V 256->1024 при K 32->128. Тогда hidden 512->2048 чтобы сохр. парам/бит

## 3. Датасет
- Train / Holdout непересекаются, holdout не видеть при обучении
- Фиксировать N_BASE*PER_BASE = N

## 4. Модель
- Фиксировать число параметров ~ hidden*in + hidden*out
- Словарь/без словаря — отдельный эксперимент, не мешать

## 5. Обучение
- Одинаково epochs*steps, early stop по holdout, логировать loss/epoch

## 6. Оценка
- Point ham и ball ham на одном holdout, same seed
- 3 прогона с разными seed -> mean±std

## 7. Отчёт
- Таблица: что меняли | что фиксировали | ham point | ham ball | same | время
- Полный лог: /tmp/report.txt + код коммит

## 8. Ловушки
- Не сравнивать 15 vs 20 эпох, 80k vs 120k, 64 vs 512 hidden
- Не оценивать на train, только holdout
- Не подгонять доку под результат — писать статус
