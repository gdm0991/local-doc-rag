"""Генератор синтетической «инженерной документации» для local-doc-rag.

Все стенды, ПЛК, модули, датчики, номера протоколов и значения ВЫМЫШЛЕНЫ.
Совпадения с реальными изделиями и объектами случайны.

Генератор детерминирован (фиксированный seed): один и тот же запуск даёт
тот же корпус. Кроме документов пишется facts.json — реестр фактов с
указанием, в каких документах каждый факт упомянут. По нему строятся
вопросы для eval (eval/make_questions.py).

Запуск:
    python data/synthetic/make_synthetic.py            # в data/synthetic/docs
    python data/synthetic/make_synthetic.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

SEED = 20261001

# ---------------------------------------------------------------------------
# Справочники (вымышленные)
# ---------------------------------------------------------------------------

PLC_MODELS = {
    "К-210": {"slots": 4, "mem_kb": 256, "cycle_ms": 20, "ports": 1},
    "К-220": {"slots": 8, "mem_kb": 1024, "cycle_ms": 10, "ports": 2},
    "К-240": {"slots": 16, "mem_kb": 4096, "cycle_ms": 5, "ports": 2},
}

MODULES = {
    "МВА-8": "8 аналоговых входов 4–20 мА, разрешение 16 бит, групповая гальваноразвязка",
    "МВТ-8": "8 входов термопар (типы K, L), компенсация холодного спая встроенная",
    "МВД-16": "16 дискретных входов =24 В, время фильтрации настраивается 1–20 мс",
    "МВК-16": "16 дискретных выходов, транзисторные, до 0,5 А на канал",
    "МАВ-4": "4 аналоговых выхода 4–20 мА",
}

SENSORS = {
    "ДД-100": "датчик избыточного давления, выход 4–20 мА, класс точности 0,25",
    "ДД-250": "датчик избыточного давления высокого диапазона, выход 4–20 мА, класс точности 0,5",
    "ТП-К": "термопара типа K (ХА), кабельная, диаметр 3 мм",
    "ДЧ-3": "датчик частоты вращения с преобразователем в 4–20 мА",
    "РМ-40": "расходомер с выходом 4–20 мА, погрешность 1 %",
    "ТТ-5": "преобразователь тока с выходом 4–20 мА",
    "ДВ-20": "датчик виброскорости с выходом 4–20 мА, диапазон 0–20 мм/с",
    "ДМ-10": "датчик крутящего момента с выходом 4–20 мА",
}

# Каталог параметров: тег -> (наименование, ед., диапазон, датчик, модуль, направление уставки)
PARAMS = {
    "P_OIL": ("Давление масла на выходе насоса", "МПа", (0, 1.6), "ДД-100", "МВА-8", "min"),
    "P_OIL_IN": ("Давление масла на входе", "МПа", (0, 1.0), "ДД-100", "МВА-8", "min"),
    "T_OIL": ("Температура масла в баке", "°C", (0, 150), "ТП-К", "МВТ-8", "max"),
    "N_ROT": ("Частота вращения вала", "об/мин", (0, 12000), "ДЧ-3", "МВА-8", "max"),
    "Q_OIL": ("Расход масла", "л/мин", (0, 200), "РМ-40", "МВА-8", "min"),
    "I_MOT": ("Ток двигателя привода", "А", (0, 250), "ТТ-5", "МВА-8", "max"),
    "V_BRG": ("Виброскорость опоры подшипника", "мм/с", (0, 20), "ДВ-20", "МВА-8", "max"),
    "P_FUEL": ("Давление топлива на входе регулятора", "МПа", (0, 10), "ДД-250", "МВА-8", "max"),
    "Q_FUEL": ("Расход топлива", "л/мин", (0, 60), "РМ-40", "МВА-8", "max"),
    "T_FUEL": ("Температура топлива", "°C", (0, 120), "ТП-К", "МВТ-8", "max"),
    "M_TORQ": ("Крутящий момент на входном валу", "Н·м", (0, 2000), "ДМ-10", "МВА-8", "max"),
    "T_GEAR": ("Температура масла в картере редуктора", "°C", (0, 150), "ТП-К", "МВТ-8", "max"),
    "T_W_IN": ("Температура воды на входе теплообменника", "°C", (0, 100), "ТП-К", "МВТ-8", "max"),
    "T_W_OUT": ("Температура воды на выходе теплообменника", "°C", (0, 100), "ТП-К", "МВТ-8", "max"),
    "P_WATER": ("Давление воды в контуре", "МПа", (0, 1.6), "ДД-100", "МВА-8", "max"),
    "P_HYD": ("Давление в гидросистеме", "МПа", (0, 40), "ДД-250", "МВА-8", "max"),
    "P_TEST": ("Испытательное давление в изделии", "МПа", (0, 60), "ДД-250", "МВА-8", "max"),
    "T_HYD": ("Температура рабочей жидкости", "°C", (0, 120), "ТП-К", "МВТ-8", "max"),
    "T_WIND": ("Температура обмотки статора", "°C", (0, 200), "ТП-К", "МВТ-8", "max"),
    "U_MOT": ("Напряжение питания двигателя", "В", (0, 500), "ТТ-5", "МВА-8", "max"),
    "A_VIB": ("Виброускорение стола", "м/с²", (0, 200), "ДВ-20", "МВА-8", "max"),
    "F_VIB": ("Частота возбуждения вибратора", "Гц", (0, 2000), "ДЧ-3", "МВА-8", "max"),
}

# Типовые значения уставок (предупредительная, аварийная) для каждого тега.
NOMINAL_SETPOINTS = {
    "P_OIL": (0.30, 0.25), "P_OIL_IN": (0.08, 0.05), "T_OIL": (85, 95), "N_ROT": (9500, 10200),
    "Q_OIL": (40, 30), "I_MOT": (180, 210), "V_BRG": (7.1, 11.2), "P_FUEL": (7.5, 8.2),
    "Q_FUEL": (45, 52), "T_FUEL": (70, 80), "M_TORQ": (1600, 1800), "T_GEAR": (90, 105),
    "T_W_IN": (60, 70), "T_W_OUT": (80, 90), "P_WATER": (1.0, 1.2), "P_HYD": (28, 32),
    "P_TEST": (45, 50), "T_HYD": (60, 70), "T_WIND": (130, 155), "U_MOT": (420, 440),
    "A_VIB": (150, 180), "F_VIB": (1800, 1950),
}

STANDS = [
    {"code": "ИС-3", "slug": "is3", "object": "масляных насосов", "plc": "К-210",
     "params": ["P_OIL", "P_OIL_IN", "T_OIL", "N_ROT", "Q_OIL", "I_MOT"]},
    {"code": "ИС-5", "slug": "is5", "object": "топливных регуляторов", "plc": "К-220",
     "params": ["P_FUEL", "Q_FUEL", "T_FUEL", "N_ROT", "P_OIL", "I_MOT", "V_BRG"]},
    {"code": "ИС-7", "slug": "is7", "object": "редукторов", "plc": "К-240",
     "params": ["N_ROT", "M_TORQ", "T_GEAR", "P_OIL", "T_OIL", "V_BRG", "I_MOT", "Q_OIL"]},
    {"code": "ИС-9", "slug": "is9", "object": "теплообменников", "plc": "К-220",
     "params": ["T_W_IN", "T_W_OUT", "P_WATER", "T_OIL", "P_OIL", "Q_OIL"]},
    {"code": "ГС-2", "slug": "gs2", "object": "гидравлических клапанов", "plc": "К-210",
     "params": ["P_HYD", "T_HYD", "Q_OIL", "I_MOT"]},
    {"code": "ГС-4", "slug": "gs4", "object": "трубопроводов на прочность", "plc": "К-220",
     "params": ["P_TEST", "P_HYD", "T_HYD", "I_MOT"]},
    {"code": "ЭС-1", "slug": "es1", "object": "электродвигателей", "plc": "К-220",
     "params": ["I_MOT", "U_MOT", "T_WIND", "N_ROT", "V_BRG", "M_TORQ"]},
    {"code": "ВС-6", "slug": "vs6", "object": "изделий на вибропрочность", "plc": "К-240",
     "params": ["A_VIB", "F_VIB", "T_OIL", "I_MOT", "P_OIL"]},
]

# Стенды, у которых протоколы испытаний пишутся в PDF (проверка ветки pypdf).
PDF_STANDS = {"is3", "is7", "gs4", "vs6"}


def fnum(x: float) -> str:
    """Число в русской записи: десятичная запятая, без лишних нулей."""
    if isinstance(x, int) or float(x).is_integer():
        return str(int(x))
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


# ---------------------------------------------------------------------------
# Модель стенда
# ---------------------------------------------------------------------------


@dataclass
class Channel:
    tag: str
    name: str
    unit: str
    rng: tuple
    sensor: str
    module: str
    slot: int
    ch: int
    kind: str  # AI / DI / DO


@dataclass
class Setpoint:
    tag: str
    name: str
    unit: str
    direction: str
    warn: float
    alarm: float
    delay_s: float
    action: str


@dataclass
class Stand:
    code: str
    slug: str
    object: str
    plc: str
    fw: str
    ip: str
    unit_id: int
    channels: list = field(default_factory=list)
    setpoints: list = field(default_factory=list)
    calib_months: dict = field(default_factory=dict)
    filter_hours: int = 0
    min_oil_temp: int = 0
    warmup_min: int = 0
    restart_wait_min: int = 0
    estop_channel: str = ""
    versions: list = field(default_factory=list)
    protocols: list = field(default_factory=list)
    serial_prefix: str = ""


class FactRegistry:
    """Реестр фактов: ключ -> значение и список документов, где факт упомянут."""

    def __init__(self) -> None:
        self.facts: dict[str, dict] = {}

    def put(self, key: str, value, doc: str, **meta) -> None:
        f = self.facts.setdefault(key, {"value": value, "docs": [], **meta})
        if f["value"] != value:
            raise ValueError(f"Несогласованный факт {key}: {f['value']} != {value}")
        if doc not in f["docs"]:
            f["docs"].append(doc)


def jitter(rnd: random.Random, value: float, rel: float = 0.06) -> float:
    """Разброс уставки от типового значения с округлением до «инженерного» шага."""
    v = value * (1 + rnd.uniform(-rel, rel))
    if value < 2:
        return round(v, 2)
    if value < 20:
        return round(v, 1)
    if value < 300:
        return float(round(v))
    return float(round(v / 50) * 50)


def build_stand(rnd: random.Random, spec: dict, idx: int) -> Stand:
    st = Stand(
        code=spec["code"], slug=spec["slug"], object=spec["object"], plc=spec["plc"],
        fw=f"{rnd.randint(2, 4)}.{rnd.randint(0, 9)}.{rnd.randint(1, 30)}",
        ip=f"192.168.{10 + idx}.{rnd.choice([10, 20, 50, 100])}",
        unit_id=rnd.randint(1, 20),
        serial_prefix=f"{rnd.randint(1, 9)}{rnd.randint(0, 9)}",
    )
    # Аналоговые каналы: МВА-8 начиная со слота 2, МВТ-8 в отдельном слоте.
    ai = [t for t in spec["params"] if PARAMS[t][4] == "МВА-8"]
    tc = [t for t in spec["params"] if PARAMS[t][4] == "МВТ-8"]
    offset = rnd.randint(0, 2)  # первые каналы иногда резерв
    for i, tag in enumerate(ai):
        name, unit, rng, sensor, module, _ = PARAMS[tag]
        n = i + offset
        st.channels.append(Channel(tag, name, unit, rng, sensor, module, 2 + n // 8, n % 8 + 1, "AI"))
    tc_slot = 2 + (len(ai) + offset - 1) // 8 + 1
    for i, tag in enumerate(tc):
        name, unit, rng, sensor, module, _ = PARAMS[tag]
        st.channels.append(Channel(tag, name, unit, rng, sensor, module, tc_slot, i + 1, "AI"))
    di_slot = tc_slot + 1
    estop_ch = rnd.randint(1, 4)
    st.estop_channel = f"слот {di_slot}, канал {estop_ch}"
    di = [("ESTOP", "Кнопка аварийного останова", estop_ch),
          ("DOOR_OK", "Дверь бокса закрыта", estop_ch + 1),
          ("DRV_RUN", "Привод в работе", estop_ch + 2)]
    for tag, name, ch in di:
        st.channels.append(Channel(tag, name, "—", (0, 1), "—", "МВД-16", di_slot, ch, "DI"))
    for tag, name, ch in [("DRV_START", "Команда пуска привода", 1), ("V_DUMP", "Клапан сброса давления", 2),
                          ("LAMP_ALM", "Лампа «Авария»", 3)]:
        st.channels.append(Channel(tag, name, "—", (0, 1), "—", "МВК-16", di_slot + 1, ch, "DO"))

    for tag in spec["params"]:
        name, unit, rng, sensor, module, direction = PARAMS[tag]
        w, a = NOMINAL_SETPOINTS[tag]
        warn, alarm = jitter(rnd, w), jitter(rnd, a)
        # Сохраняем порядок: для max аварийная выше предупредительной, для min — ниже.
        if direction == "max" and alarm <= warn:
            alarm = jitter(rnd, warn * 1.1, 0.0)
        if direction == "min" and alarm >= warn:
            alarm = jitter(rnd, warn * 0.85, 0.0)
        st.setpoints.append(Setpoint(tag, name, unit, direction, warn, alarm,
                                     rnd.choice([0.5, 1, 2, 3, 5]),
                                     rnd.choice(["останов привода", "останов привода и сброс давления",
                                                 "плавное снижение режима до малого газа"
                                                 if tag in ("N_ROT", "M_TORQ") else "останов привода"])))
    st.calib_months = {
        "давления": rnd.choice([6, 12]),
        "температуры": rnd.choice([12, 24]),
        "прочих аналоговых": rnd.choice([12, 24]),
    }
    st.filter_hours = rnd.choice([250, 300, 500, 750, 1000])
    st.min_oil_temp = rnd.choice([15, 20, 25, 30])
    st.warmup_min = rnd.choice([5, 10, 15, 20])
    st.restart_wait_min = rnd.choice([5, 10, 15, 30])

    # Журнал версий ПО ПЛК. Последнее изменение задержки согласовано с картой уставок.
    n_ver = rnd.randint(3, 5)
    years = sorted(rnd.sample(range(2021, 2026), k=min(n_ver, 5)))
    changes_pool = [
        "добавлен архив аварийных событий на 500 записей",
        "исправлена ошибка масштабирования канала при обрыве линии",
        "добавлен контроль связи с панелью оператора (сторожевой таймер 3 с)",
        "введена блокировка пуска при открытой двери бокса",
        "добавлена запись трендов с периодом 100 мс",
        "переработана логика квитирования аварий",
        "увеличено число сохраняемых протоколов испытаний до 200",
    ]
    rnd.shuffle(changes_pool)
    sp_changed = rnd.choice(st.setpoints)
    old_delay = rnd.choice([d for d in [0.5, 1, 2, 3, 5] if d != sp_changed.delay_s])
    change_at = rnd.randint(1, n_ver - 1)
    for i in range(n_ver):
        ver = f"1.{i}"
        date = f"{rnd.randint(1, 28):02d}.{rnd.randint(1, 12):02d}.{years[i]}"
        items = ["первая редакция программы"] if i == 0 else [changes_pool[i - 1]]
        if i == change_at:
            items.append(f"задержка аварийной уставки «{sp_changed.name}» изменена с "
                         f"{fnum(old_delay)} до {fnum(sp_changed.delay_s)} с")
        st.versions.append({"ver": ver, "date": date, "items": items})
    st.delay_change = {"tag": sp_changed.tag, "name": sp_changed.name, "ver": st.versions[change_at]["ver"],
                       "old": old_delay, "new": sp_changed.delay_s}

    # Протоколы испытаний: два на стенд, один из них может быть «не соответствует».
    main_sp = st.setpoints[0]
    for k in range(2):
        year = rnd.choice([2024, 2025])
        num = f"ПИ-{st.code}-{year}-{rnd.randint(1, 120):03d}"
        serial = f"{st.serial_prefix}{rnd.randint(100, 999)}"
        modes = []
        fail = (k == 1 and rnd.random() < 0.6)
        lo, hi = PARAMS[main_sp.tag][2]
        for m in range(1, 4):
            norm_lo = round(lo + (hi - lo) * (0.2 + 0.15 * m), 2 if hi < 20 else 0)
            norm_hi = round(norm_lo * 1.08, 2 if hi < 20 else 0)
            meas = round(rnd.uniform(norm_lo, norm_hi), 2 if hi < 20 else 0)
            if fail and m == 3:
                meas = round(norm_hi * 1.05, 2 if hi < 20 else 0)
            modes.append({"mode": m, "norm_lo": norm_lo, "norm_hi": norm_hi, "meas": meas})
        st.protocols.append({
            "num": num, "date": f"{rnd.randint(1, 28):02d}.{rnd.randint(1, 12):02d}.{year}",
            "serial": serial, "param": main_sp.name, "unit": main_sp.unit, "tag": main_sp.tag,
            "modes": modes,
            "result": "не соответствует" if fail else "соответствует",
            "duration_min": rnd.choice([30, 45, 60, 90, 120]),
        })
    return st


# ---------------------------------------------------------------------------
# Генерация документов
# ---------------------------------------------------------------------------

DISCLAIMER = ("> Синтетический документ для демонстрации local-doc-rag. Стенд, оборудование и значения "
              "вымышлены.\n")


def sp_word(sp: Setpoint) -> str:
    return "не ниже" if sp.direction == "min" else "не выше"


def doc_re(st: Stand, reg: FactRegistry, doc: str) -> str:
    plc = PLC_MODELS[st.plc]
    reg.put(f"{st.slug}.plc", st.plc, doc)
    reg.put(f"{st.slug}.ip", st.ip, doc)
    lines = [f"# Руководство по эксплуатации стенда {st.code}", "", DISCLAIMER,
             "## 1. Назначение", "",
             f"Стенд {st.code} предназначен для приёмо-сдаточных и периодических испытаний {st.object}. "
             f"Стенд эксплуатируется в закрытом боксе испытательной лаборатории.", "",
             "## 2. Состав системы управления", "",
             f"Система управления построена на программируемом логическом контроллере {st.plc} серии К-200 "
             f"(версия встроенного ПО {st.fw}). Контроллер {st.plc} имеет {plc['slots']} слотов расширения "
             f"и минимальное время цикла {plc['cycle_ms']} мс.", "",
             f"Контроллер подключён к сети стенда по адресу {st.ip}, обмен с компьютером оператора — "
             f"Modbus TCP, порт 502, адрес устройства (Unit ID) {st.unit_id}.", "",
             "Модули ввода-вывода:", ""]
    mods = sorted({(c.slot, c.module) for c in st.channels})
    for slot, mod in mods:
        lines.append(f"- слот {slot}: {mod} — {MODULES[mod]};")
    lines += ["", "## 3. Контролируемые параметры", "",
              f"На стенде {st.code} контролируются {len(st.setpoints)} аналоговых параметров:", ""]
    for sp in st.setpoints:
        lines.append(f"- {sp.name.lower()}, {sp.unit};")
    lines += ["", "## 4. Защиты", "",
              "При выходе параметра за аварийную уставку ПЛК выполняет защитное действие. "
              "Основные защиты стенда:", ""]
    for sp in st.setpoints[:3]:
        lines.append(f"- {sp.name.lower()}: аварийная уставка {fnum(sp.alarm)} {sp.unit} "
                     f"(срабатывание {'при снижении' if sp.direction == 'min' else 'при превышении'}), "
                     f"действие — {sp.action};")
        reg.put(f"{st.slug}.{sp.tag}.alarm", fnum(sp.alarm), doc, unit=sp.unit, name=sp.name)
    lines += ["", "Полный перечень уставок приведён в документе «Карта уставок стенда "
              f"{st.code}».", "",
              "## 5. Режимы работы", "",
              "Предусмотрены режимы «Наладка», «Испытание по программе» и «Ручное управление». "
              "Переключение режимов выполняется только при остановленном приводе.", ""]
    return "\n".join(lines)


def doc_signals(st: Stand, reg: FactRegistry, doc: str) -> str:
    lines = [f"# Перечень входных и выходных сигналов стенда {st.code}", "", DISCLAIMER,
             f"Контроллер: {st.plc}. Нумерация каналов в модуле — с 1.", "",
             "| № | Тег | Наименование | Тип | Модуль | Слот | Канал | Диапазон | Ед. изм. | Датчик |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(st.channels, 1):
        rng = f"{fnum(c.rng[0])}…{fnum(c.rng[1])}" if c.kind == "AI" else "0/1"
        lines.append(f"| {i} | {c.tag} | {c.name} | {c.kind} | {c.module} | {c.slot} | {c.ch} | {rng} | "
                     f"{c.unit} | {c.sensor} |")
        if c.kind == "AI":
            reg.put(f"{st.slug}.{c.tag}.channel", f"слот {c.slot}, канал {c.ch}", doc, name=c.name,
                    module=c.module)
            reg.put(f"{st.slug}.{c.tag}.sensor", c.sensor, doc, name=c.name)
    reg.put(f"{st.slug}.estop", st.estop_channel, doc)
    lines += ["", f"Всего сигналов: {len(st.channels)}.", ""]
    return "\n".join(lines)


def doc_setpoints(st: Stand, reg: FactRegistry, doc: str) -> str:
    lines = [f"# Карта уставок стенда {st.code}", "", DISCLAIMER,
             "Уставки хранятся в энергонезависимой памяти ПЛК и изменяются только с уровнем доступа «Инженер».", "",
             "| Тег | Параметр | Ед. | Предупредительная | Аварийная | Задержка, с | Действие при аварии |",
             "|---|---|---|---|---|---|---|"]
    for sp in st.setpoints:
        lines.append(f"| {sp.tag} | {sp.name} | {sp.unit} | {fnum(sp.warn)} | {fnum(sp.alarm)} | "
                     f"{fnum(sp.delay_s)} | {sp.action} |")
        reg.put(f"{st.slug}.{sp.tag}.alarm", fnum(sp.alarm), doc, unit=sp.unit, name=sp.name)
        reg.put(f"{st.slug}.{sp.tag}.warn", fnum(sp.warn), doc, unit=sp.unit, name=sp.name)
        reg.put(f"{st.slug}.{sp.tag}.delay", fnum(sp.delay_s), doc, name=sp.name)
    lines += ["",
              "Для параметров с контролем снижения («Давление масла», «Расход масла») авария формируется, "
              "когда значение становится ниже уставки; для остальных — выше.", ""]
    return "\n".join(lines)


def protocol_text(st: Stand, p: dict) -> list[str]:
    lines = [f"Протокол испытаний № {p['num']}", "",
             f"Стенд: {st.code}. Дата испытаний: {p['date']}.",
             f"Объект испытаний: изделие заводской № {p['serial']}.",
             f"Контролируемый параметр: {p['param']}, {p['unit']}.",
             f"Продолжительность испытаний: {p['duration_min']} мин.", "",
             "Результаты по режимам:"]
    for m in p["modes"]:
        lines.append(f"Режим {m['mode']}: норма {fnum(m['norm_lo'])}…{fnum(m['norm_hi'])} {p['unit']}, "
                     f"измерено {fnum(m['meas'])} {p['unit']}.")
    lines += ["", f"Заключение: изделие {p['result']} требованиям программы испытаний."]
    if p["result"] == "не соответствует":
        lines.append("Причина: на режиме 3 измеренное значение выше верхней границы нормы. "
                     "Изделие направлено на доработку.")
    return lines


def doc_protocol_md(st: Stand, p: dict, reg: FactRegistry, doc: str) -> str:
    register_protocol(st, p, reg, doc)
    lines = protocol_text(st, p)
    return "\n".join([f"# {lines[0]}", "", DISCLAIMER] + lines[1:]) + "\n"


def register_protocol(st: Stand, p: dict, reg: FactRegistry, doc: str) -> None:
    reg.put(f"proto.{p['num']}.result", p["result"], doc, stand=st.code)
    reg.put(f"proto.{p['num']}.serial", p["serial"], doc, stand=st.code)
    reg.put(f"proto.{p['num']}.mode2", fnum(p["modes"][1]["meas"]), doc, stand=st.code, unit=p["unit"],
            param=p["param"])
    reg.put(f"proto.{p['num']}.duration", str(p["duration_min"]), doc, stand=st.code)


def doc_startup(st: Stand, reg: FactRegistry, doc: str) -> str:
    reg.put(f"{st.slug}.min_oil_temp", str(st.min_oil_temp), doc)
    reg.put(f"{st.slug}.warmup", str(st.warmup_min), doc)
    lines = [f"ИНСТРУКЦИЯ ПО ПУСКУ СТЕНДА {st.code}", "",
             "Синтетический документ для демонстрации local-doc-rag. Значения вымышлены.", "",
             "1. Проверить отсутствие посторонних лиц в боксе, закрыть дверь бокса "
             "(сигнал DOOR_OK должен быть активен).",
             f"2. Включить питание шкафа управления. Дождаться загрузки контроллера {st.plc} "
             "(индикатор RUN горит постоянно).",
             f"3. Проверить температуру масла: пуск разрешён при температуре не ниже {st.min_oil_temp} °C. "
             "При более низкой температуре включить подогрев бака.",
             "4. Выбрать на компьютере оператора программу испытаний и режим «Испытание по программе».",
             f"5. Подать команду пуска привода. Прогреть изделие на малом режиме в течение {st.warmup_min} мин.",
             "6. Убедиться, что все параметры находятся в пределах предупредительных уставок, и перейти "
             "к первому режиму программы.",
             "",
             "Запрещается пуск при активной аварии или неквитированном сообщении на экране оператора.", ""]
    return "\n".join(lines)


def doc_emergency(st: Stand, reg: FactRegistry, doc: str) -> str:
    reg.put(f"{st.slug}.restart_wait", str(st.restart_wait_min), doc)
    reg.put(f"{st.slug}.estop", st.estop_channel, doc)
    lines = [f"# Инструкция по аварийному останову стенда {st.code}", "", DISCLAIMER,
             "## Когда нажимать кнопку аварийного останова", "",
             "- при появлении дыма, течи масла или топлива;",
             "- при постороннем шуме или стуке в изделии;",
             "- при отказе защит, когда параметр вышел за аварийную уставку, а останов не произошёл.", "",
             "## Что делает система", "",
             f"Кнопка аварийного останова заведена на дискретный вход ПЛК ({st.estop_channel}, модуль МВД-16). "
             "При её нажатии контроллер снимает команду пуска привода (DRV_START), открывает клапан сброса "
             "давления (V_DUMP) и включает лампу «Авария».", "",
             "## Действия оператора после останова", "",
             "1. Не открывать дверь бокса до полной остановки вращения.",
             "2. Записать в журнал время и причину останова.",
             "3. Устранить причину, квитировать аварию на экране оператора.",
             f"4. Повторный пуск — не ранее чем через {st.restart_wait_min} мин после останова "
             "и только с разрешения начальника смены.", ""]
    return "\n".join(lines)


def doc_calibration(st: Stand, reg: FactRegistry, doc: str) -> str:
    for k, v in st.calib_months.items():
        reg.put(f"{st.slug}.calib.{k}", str(v), doc)
    lines = [f"# Методика калибровки измерительных каналов стенда {st.code}", "", DISCLAIMER,
             "## Периодичность", "",
             "| Группа каналов | Межкалибровочный интервал, мес |", "|---|---|"]
    for k, v in st.calib_months.items():
        lines.append(f"| Каналы {k} | {v} |")
    lines += ["", "## Средства калибровки", "",
              "- калибратор токовой петли КТ-7 (вымышленный), погрешность ±0,02 % от диапазона;",
              "- калибратор температуры КТМ-3 (вымышленный) для каналов термопар.", "",
              "## Порядок", "",
              "1. Отключить датчик, подключить калибратор к входу модуля.",
              "2. Подать 5 точек: 0, 25, 50, 75 и 100 % диапазона.",
              "3. Допустимая приведённая погрешность канала — не более 0,5 %.",
              "4. Результат записать в паспорт канала; при превышении погрешности канал выводится из работы.", ""]
    return "\n".join(lines)


def doc_changelog(st: Stand, reg: FactRegistry, doc: str) -> str:
    dc = st.delay_change
    reg.put(f"{st.slug}.delay_change_ver", dc["ver"], doc, name=dc["name"], old=fnum(dc["old"]),
            new=fnum(dc["new"]))
    reg.put(f"{st.slug}.last_ver", st.versions[-1]["ver"], doc, date=st.versions[-1]["date"])
    lines = [f"# Журнал изменений программы ПЛК стенда {st.code}", "", DISCLAIMER,
             "| Версия | Дата | Изменения |", "|---|---|---|"]
    for v in st.versions:
        lines.append(f"| {v['ver']} | {v['date']} | {'; '.join(v['items'])} |")
    lines += ["", f"Актуальная версия: {st.versions[-1]['ver']}.", ""]
    return "\n".join(lines)


def doc_maintenance(st: Stand, reg: FactRegistry, doc: str) -> str:
    reg.put(f"{st.slug}.filter_hours", str(st.filter_hours), doc)
    lines = [f"# Регламент технического обслуживания стенда {st.code}", "", DISCLAIMER,
             "| Периодичность | Работа |", "|---|---|",
             "| Ежедневно | Проверка уровня масла в баке, осмотр на отсутствие течей |",
             "| Еженедельно | Проверка срабатывания кнопки аварийного останова |",
             f"| Каждые {st.filter_hours} моточасов | Замена фильтроэлемента маслосистемы |",
             "| Ежемесячно | Проверка затяжки клеммных соединений в шкафу управления |",
             "| Ежегодно | Проверка сопротивления изоляции силовых цепей |",
             "", "Учёт моточасов ведёт ПЛК, счётчик отображается на экране «Сервис».", ""]
    return "\n".join(lines)


def doc_modbus(st: Stand, reg: FactRegistry, doc: str) -> str:
    lines = [f"# Карта регистров Modbus TCP стенда {st.code}", "", DISCLAIMER,
             f"Адрес контроллера {st.ip}, порт 502, Unit ID {st.unit_id}. Значения аналоговых параметров — "
             "REAL (2 регистра, порядок слов big-endian).", "",
             "| Тег | Параметр | Область | Адрес | Тип |", "|---|---|---|---|---|"]
    reg.put(f"{st.slug}.unit_id", str(st.unit_id), doc)
    reg.put(f"{st.slug}.ip", st.ip, doc)
    ai = [c for c in st.channels if c.kind == "AI"]
    for i, c in enumerate(ai):
        addr = 30001 + 2 * i
        lines.append(f"| {c.tag} | {c.name} | Input Registers | {addr} | REAL |")
        reg.put(f"{st.slug}.{c.tag}.modbus", str(addr), doc, name=c.name)
    for i, sp in enumerate(st.setpoints):
        lines.append(f"| {sp.tag}_ALM | Аварийная уставка: {sp.name.lower()} | Holding Registers | "
                     f"{40001 + 2 * i} | REAL |")
    lines.append("")
    return "\n".join(lines)


def common_docs(reg: FactRegistry) -> dict[str, str]:
    docs: dict[str, str] = {}
    d = "obshchie/k200_rukovodstvo_programmista.md"
    rows = []
    for m, p in PLC_MODELS.items():
        rows.append(f"| {m} | {p['slots']} | {p['mem_kb']} | {p['cycle_ms']} | {p['ports']} |")
        reg.put(f"plc.{m}.slots", str(p["slots"]), d)
        reg.put(f"plc.{m}.mem", str(p["mem_kb"]), d)
    errors = [("E01", "нет связи с модулем расширения"), ("E04", "ошибка контрольной суммы программы"),
              ("E07", "переполнение времени цикла"), ("E12", "обрыв линии аналогового входа (ток < 3,6 мА)"),
              ("E13", "превышение тока аналогового входа (ток > 21 мА)"), ("E20", "разряд батареи часов"),
              ("E31", "конфликт адресов Modbus")]
    for code, text in errors:
        reg.put(f"plc.err.{code}", text, d)
    reg.put("plc.modbus_conn", "8", d)
    docs[d] = "\n".join([
        "# Контроллеры серии К-200. Руководство программиста", "", DISCLAIMER,
        "## Модели", "",
        "| Модель | Слотов расширения | Память программы, КБ | Мин. время цикла, мс | Портов Ethernet |",
        "|---|---|---|---|---|", *rows, "",
        "## Языки программирования", "",
        "Поддерживаются ST и FBD. Программа загружается через порт Ethernet.", "",
        "## Modbus TCP", "",
        "Встроенный сервер Modbus TCP, порт по умолчанию 502, одновременно не более 8 клиентских подключений.",
        "", "## Коды ошибок", "", "| Код | Описание |", "|---|---|",
        *[f"| {c} | {t} |" for c, t in errors], ""])

    d = "obshchie/moduli_k200.md"
    for m, t in MODULES.items():
        reg.put(f"mod.{m}", t, d)
    docs[d] = "\n".join(["# Модули ввода-вывода серии К-200", "", DISCLAIMER,
                         "| Модуль | Характеристики |", "|---|---|",
                         *[f"| {m} | {t} |" for m, t in MODULES.items()], "",
                         "Модули устанавливаются в слоты расширения контроллера, горячая замена не поддерживается.",
                         ""])

    d = "obshchie/spravochnik_datchikov.md"
    for s, t in SENSORS.items():
        reg.put(f"sens.{s}", t, d)
    docs[d] = "\n".join(["# Справочник датчиков испытательных стендов", "", DISCLAIMER,
                         *[f"- **{s}** — {t}." for s, t in SENSORS.items()], ""])

    d = "obshchie/glossariy.md"
    docs[d] = "\n".join(["# Глоссарий", "", DISCLAIMER,
                         "- **Предупредительная уставка** — граница, при переходе которой оператор получает "
                         "сообщение, но испытание продолжается.",
                         "- **Аварийная уставка** — граница, при переходе которой ПЛК выполняет защитное действие.",
                         "- **Задержка срабатывания** — время, в течение которого параметр должен непрерывно "
                         "находиться за уставкой, прежде чем сработает защита.",
                         "- **Квитирование** — подтверждение оператором того, что он увидел аварийное сообщение.",
                         "- **Моточасы** — наработка привода стенда, учитываемая ПЛК.", ""])

    d = "obshchie/trebovaniya_bezopasnosti.md"
    reg.put("safety.min_staff", "2", d)
    docs[d] = "\n".join(["# Общие требования безопасности при работе на испытательных стендах", "", DISCLAIMER,
                         "1. К работе на стендах допускаются лица, прошедшие инструктаж и стажировку.",
                         "2. Испытания проводятся сменой не менее 2 человек: оператор и наблюдающий.",
                         "3. Нахождение в боксе во время работы привода запрещено.",
                         "4. Изменение уставок защит допускается только по письменному распоряжению.",
                         "5. Средства пожаротушения проверяются перед началом каждой смены.", ""])
    return docs


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def write_pdf(path: Path, lines: list[str]) -> bool:
    """Пишет простой PDF с кириллицей. Возвращает False, если нет reportlab или шрифта."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError:
        return False
    font = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
    if font is None:
        return False
    pdfmetrics.registerFont(TTFont("DocFont", font))
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(lines[0])
    y = 800
    for i, line in enumerate(lines):
        c.setFont("DocFont", 13 if i == 0 else 10)
        c.drawString(50, y, line)
        y -= 18 if i == 0 else 14
    c.save()
    return True


# ---------------------------------------------------------------------------


def generate(out: Path) -> dict:
    rnd = random.Random(SEED)
    reg = FactRegistry()
    stands = [build_stand(rnd, spec, i) for i, spec in enumerate(STANDS)]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    written = 0
    pdf_fallback = []

    def write(rel: str, text: str) -> None:
        nonlocal written
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        written += 1

    for rel, text in common_docs(reg).items():
        write(rel, text)

    for st in stands:
        s = st.slug
        write(f"{s}/re_{s}.md", doc_re(st, reg, f"{s}/re_{s}.md"))
        write(f"{s}/perechen_signalov_{s}.md", doc_signals(st, reg, f"{s}/perechen_signalov_{s}.md"))
        write(f"{s}/karta_ustavok_{s}.md", doc_setpoints(st, reg, f"{s}/karta_ustavok_{s}.md"))
        write(f"{s}/instrukciya_pusk_{s}.txt", doc_startup(st, reg, f"{s}/instrukciya_pusk_{s}.txt"))
        write(f"{s}/instrukciya_avariynyy_ostanov_{s}.md",
              doc_emergency(st, reg, f"{s}/instrukciya_avariynyy_ostanov_{s}.md"))
        write(f"{s}/metodika_kalibrovki_{s}.md", doc_calibration(st, reg, f"{s}/metodika_kalibrovki_{s}.md"))
        write(f"{s}/zhurnal_po_plk_{s}.md", doc_changelog(st, reg, f"{s}/zhurnal_po_plk_{s}.md"))
        write(f"{s}/reglament_to_{s}.md", doc_maintenance(st, reg, f"{s}/reglament_to_{s}.md"))
        write(f"{s}/modbus_karta_{s}.md", doc_modbus(st, reg, f"{s}/modbus_karta_{s}.md"))
        for k, p in enumerate(st.protocols, 1):
            if s in PDF_STANDS:
                rel = f"{s}/protokol_{k}_{s}.pdf"
                (out / s).mkdir(parents=True, exist_ok=True)
                lines = protocol_text(st, p) + ["", "Синтетический документ. Значения вымышлены."]
                if write_pdf(out / rel, lines):
                    register_protocol(st, p, reg, rel)
                    written += 1
                    continue
                pdf_fallback.append(rel)
            rel = f"{s}/protokol_{k}_{s}.md"
            write(rel, doc_protocol_md(st, p, reg, rel))

    facts = {"seed": SEED, "documents": written, "pdf_fallback_to_md": pdf_fallback,
             "stands": [{"code": st.code, "slug": st.slug, "object": st.object,
                         "params": [sp.tag for sp in st.setpoints],
                         "protocols": [p["num"] for p in st.protocols]} for st in stands],
             "facts": reg.facts}
    (out.parent / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    return facts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "docs")
    args = ap.parse_args()
    facts = generate(args.out)
    print(f"Документов: {facts['documents']}, фактов в реестре: {len(facts['facts'])}")
    if facts["pdf_fallback_to_md"]:
        print("ВНИМАНИЕ: PDF не созданы (нет reportlab или шрифта), записаны как .md:",
              ", ".join(facts["pdf_fallback_to_md"]))


if __name__ == "__main__":
    main()
