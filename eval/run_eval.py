"""Оценка local-doc-rag на синтетическом корпусе.

Что считается (определения — те же, что в README):
  Recall@5  — доля вопросов с ответом, у которых хотя бы один из первых 5 фрагментов
              выдачи взят из документа-источника (поле sources).
  MRR@10    — среднее 1/r, где r — место первого фрагмента из документа-источника
              в первых 10 фрагментах (0, если не попал).
  Отказы    — доля «не знаю» на ловушках (ответа в корпусе нет) и, отдельно,
              доля ложных «не знаю» на вопросах с ответом.
  Латентность — время вызова pipeline.search / pipeline.ask в процессе
              (time.perf_counter), без HTTP. Кеш эмбеддингов запросов очищается
              перед каждым замером. Индексация и загрузка модели в замер не входят.

Запуск:
  python eval/run_eval.py                                  # только поиск, без LLM
  python eval/run_eval.py --llm llamacpp --model path.gguf # плюс ответы локальной модели
  python eval/run_eval.py --calibrate                      # подобрать порог отказа на calibration.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_doc_rag.llm import make_generator  # noqa: E402
from local_doc_rag.pipeline import MODES, RagPipeline  # noqa: E402
from local_doc_rag.text import normalize  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(values: list[float], q: float) -> float:
    """Перцентиль методом ближайшего ранга (без интерполяции)."""
    s = sorted(values)
    if not s:
        return float("nan")
    idx = max(0, min(len(s) - 1, math.ceil(q / 100 * len(s)) - 1))
    return s[idx]


def first_hit_rank(results: list[dict], sources: list[str]) -> int | None:
    for r, item in enumerate(results, 1):
        if item["doc_id"] in sources:
            return r
    return None


def timed_search(pipe: RagPipeline, q: str, k: int, mode: str) -> tuple[list[dict], float]:
    pipe.vec._qcache.clear()
    t = time.perf_counter()
    res = pipe.search(q, k=k, mode=mode)
    return res, (time.perf_counter() - t) * 1000


def eval_retrieval(pipe: RagPipeline, questions: list[dict]) -> dict:
    ans = [q for q in questions if q["kind"] == "answerable"]
    out = {}
    for mode in MODES:
        hits5, rr, lat, per_q = 0, [], [], {}
        for q in ans:
            res, ms = timed_search(pipe, q["question"], 10, mode)
            lat.append(ms)
            r = first_hit_rank(res, q["sources"])
            hits5 += int(r is not None and r <= 5)
            rr.append(1 / r if r else 0.0)
            per_q[q["id"]] = r
        out[mode] = {"recall@5": hits5 / len(ans), "mrr@10": statistics.mean(rr),
                     "p50_ms": statistics.median(lat), "p95_ms": pct(lat, 95), "ranks": per_q, "n": len(ans)}
    return out


def eval_refusal_nollm(pipe: RagPipeline, questions: list[dict]) -> dict:
    traps = [q for q in questions if q["kind"] == "trap"]
    ans = [q for q in questions if q["kind"] == "answerable"]
    rows = {}
    for q in questions:
        r = pipe.ask(q["question"], k=5, use_llm=False)
        rows[q["id"]] = {"refused": r["refused"], "coverage": r["coverage"]}
    return {"threshold": pipe.coverage_threshold,
            "trap_refusal": sum(rows[q["id"]]["refused"] for q in traps) / len(traps),
            "false_refusal": sum(rows[q["id"]]["refused"] for q in ans) / len(ans),
            "n_traps": len(traps), "n_answerable": len(ans), "rows": rows}


def calibrate(pipe: RagPipeline, calib: list[dict]) -> dict:
    cov = [(pipe.ask(q["question"], k=5, use_llm=False)["coverage"], q["kind"]) for q in calib]
    best = None
    # Кандидаты — середины между соседними значениями покрытия: порог не прилипает к точке набора.
    vals = sorted({c for c, _ in cov})
    cands = [0.0] + [round((a + b) / 2, 3) for a, b in zip(vals, vals[1:])] + [1.01]
    for t in cands:
        correct = sum((c < t) == (kind == "trap") for c, kind in cov)
        if best is None or correct > best[1]:
            best = (t, correct)
    return {"coverage": cov, "best_threshold": best[0], "correct": best[1], "n": len(cov)}


CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def answer_ok(answer: str, keys: list[str]) -> bool:
    """Все ключевые значения есть в ответе.

    Ссылки вида [5] удаляются до проверки, числа сравниваются по границам:
    первая версия проверки засчитала ответ «[5]» на вопрос с эталоном «5 с» (прогон 1, Q07).
    """
    a = CITATION_RE.sub(" ", normalize(answer))
    for k in keys:
        k = normalize(k)
        if re.fullmatch(r"[\d,.]+", k):
            if not re.search(rf"(?<![\d,]){re.escape(k)}(?![\d]|,\d)", a):
                return False
        elif k not in a:
            return False
    return True


def summarize_llm(rows: dict, lat: list[float], questions: list[dict], name: str) -> dict:
    traps = [q for q in questions if q["kind"] == "trap"]
    ans = [q for q in questions if q["kind"] == "answerable"]
    return {"llm": name,
            "answer_accuracy": sum(rows[q["id"]]["ok"] for q in ans) / len(ans),
            "trap_refusal": sum(rows[q["id"]]["refused"] for q in traps) / len(traps),
            "false_refusal": sum(rows[q["id"]]["refused"] for q in ans) / len(ans),
            "p50_ms": statistics.median(lat), "p95_ms": pct(lat, 95), "rows": rows}


def judge(q: dict, answer: str, refused: bool) -> bool:
    if q["kind"] == "answerable":
        return (not refused) and answer_ok(answer, q["answer_keys"])
    return refused


def eval_llm(pipe: RagPipeline, questions: list[dict]) -> dict:
    rows, lat = {}, []
    for q in questions:
        pipe.vec._qcache.clear()
        t = time.perf_counter()
        r = pipe.ask(q["question"], k=5, use_llm=True)
        ms = (time.perf_counter() - t) * 1000
        lat.append(ms)
        ok = judge(q, r["answer"], r["refused"])
        rows[q["id"]] = {"answer": r["answer"], "refused": r["refused"], "ok": ok, "ms": round(ms)}
        print(f"  {q['id']} {'OK ' if ok else 'ERR'} {ms / 1000:5.1f} с  {r['answer'][:90]!r}", flush=True)
    return summarize_llm(rows, lat, questions, pipe.generator.name)


def rescore(json_path: Path, questions: list[dict], note: str = "") -> None:
    """Пересчитать метрики LLM по сохранённым ответам (без повторной генерации) и переписать отчёт."""
    ctx = json.loads(json_path.read_text(encoding="utf-8"))
    qs = {q["id"]: q for q in questions}
    rows = ctx["llm"]["rows"]
    for qid, row in rows.items():
        row["ok"] = judge(qs[qid], row["answer"], row["refused"])
    lat = [row["ms"] for row in rows.values()]
    old = ctx["llm"]
    ctx["llm"] = summarize_llm(rows, lat, questions, old["llm"])
    # Латентность берём из исходного прогона (там точные значения, в rows — округлённые до мс)
    ctx["llm"]["p50_ms"], ctx["llm"]["p95_ms"] = old["p50_ms"], old["p95_ms"]
    ctx["rescored"] = dt.date.today().isoformat()
    if note:
        ctx["note"] = note
    ctx["questions"] = questions
    write_report(json_path.with_suffix(".md"), ctx)
    json_path.write_text(json.dumps({k: v for k, v in ctx.items() if k != "questions"}, ensure_ascii=False,
                                    indent=1, default=str), encoding="utf-8")
    print(f"Пересчитано: точность {ctx['llm']['answer_accuracy']:.2f} (было {old['answer_accuracy']:.2f})")


def env_info() -> dict:
    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem = ""
    try:
        kb = int(next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")).split()[1])
        mem = f"{kb / 1024 / 1024:.1f} ГБ"
    except (OSError, StopIteration):
        pass
    pkgs = {}
    for p in ["fastembed", "onnxruntime", "rank-bm25", "scikit-learn", "llama_cpp_python", "fastapi"]:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = "нет"
    return {"os": platform.platform(), "python": platform.python_version(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "ram": mem, "packages": pkgs}


def f2(x: float) -> str:
    return f"{x:.2f}"


def write_report(path: Path, ctx: dict) -> None:
    r, nl, env, info = ctx["retrieval"], ctx["nollm"], ctx["env"], ctx["info"]
    qs = {q["id"]: q for q in ctx["questions"]}
    L = [f"# Результаты eval — {ctx['date']}", "",
         "Файл сгенерирован `eval/run_eval.py`. Цифры ниже — один прогон, не среднее по нескольким.", "",
         *([ctx["note"], ""] if ctx.get("note") else []),
         *([f"Метрики ответов LLM пересчитаны {ctx['rescored']} исправленной проверкой "
            "(`--rescore`) по сохранённым ответам, без повторной генерации.", ""] if ctx.get("rescored") else []),
         "## Стенд замера", "",
         "| Что | Значение |", "|---|---|",
         f"| ОС | {env['os']} |", f"| CPU | {env['cpu']}, логических ядер: {env['logical_cpus']} |",
         f"| ОЗУ | {env['ram']} |", "| GPU | нет (замер только на CPU) |",
         f"| Python | {env['python']} |",
         f"| Пакеты | {', '.join(f'{k} {v}' for k, v in env['packages'].items())} |",
         f"| Векторный бэкенд | {info['embed_backend']} |",
         f"| Корпус | {info['documents']} документов, {info['chunks']} фрагментов (800 символов, перекрытие 150) |",
         f"| Индексация | {ctx['ingest']['seconds']} с (включая расчёт эмбеддингов, без загрузки модели из сети) |",
         f"| Вопросы | {r['hybrid']['n']} с ответом + {nl['n_traps']} ловушек, `eval/questions.jsonl` |",
         "", "## Поиск (вопросы с ответом)", "",
         "| Режим | Recall@5 | MRR@10 | p50, мс | p95, мс |", "|---|---|---|---|---|"]
    for m in MODES:
        x = r[m]
        L.append(f"| {m} | {f2(x['recall@5'])} | {f2(x['mrr@10'])} | {x['p50_ms']:.1f} | {x['p95_ms']:.1f} |")
    L += ["", "Латентность поиска — время `pipeline.search(k=10)` в процессе, кеш эмбеддингов запроса очищен, "
          "p95 — ближайший ранг по 30 замерам.", "",
          "## Отказ «не знаю» без LLM (режим --no-llm)", "",
          f"Правило: отказ, если покрытие запроса первым фрагментом (доля IDF-веса терминов запроса) ниже "
          f"{nl['threshold']}. Порог подобран на `eval/calibration.jsonl` (5 + 5 вопросов, не пересекаются "
          "с основным набором).", "",
          "| Метрика | Значение |", "|---|---|",
          f"| Доля «не знаю» на ловушках | {f2(nl['trap_refusal'])} ({round(nl['trap_refusal'] * nl['n_traps'])} из "
          f"{nl['n_traps']}) |",
          f"| Ложные «не знаю» на вопросах с ответом | {f2(nl['false_refusal'])} "
          f"({round(nl['false_refusal'] * nl['n_answerable'])} из {nl['n_answerable']}) |", ""]
    if ctx.get("calib"):
        c = ctx["calib"]
        L += [f"Калибровка: лучший порог {c['best_threshold']}, верных решений {c['correct']} из {c['n']}.", ""]
    if ctx.get("llm"):
        lm = ctx["llm"]
        n_ans = r["hybrid"]["n"]
        L += ["## Ответы локальной модели", "",
              f"Генератор: `{lm['llm']}`, temperature 0, контекст — 5 фрагментов гибридного поиска.", "",
              "| Метрика | Значение |", "|---|---|",
              f"| Точность ответа (все ключевые значения есть в ответе, без отказа) | {f2(lm['answer_accuracy'])} "
              f"({round(lm['answer_accuracy'] * n_ans)} из {n_ans}) |",
              f"| Доля «не знаю» на ловушках | {f2(lm['trap_refusal'])} ({round(lm['trap_refusal'] * nl['n_traps'])} из "
              f"{nl['n_traps']}) |",
              f"| Ложные «не знаю» на вопросах с ответом | {f2(lm['false_refusal'])} "
              f"({round(lm['false_refusal'] * n_ans)} из {n_ans}) |",
              f"| p50 полного ответа, мс | {lm['p50_ms']:.0f} |",
              f"| p95 полного ответа, мс | {lm['p95_ms']:.0f} |", "",
              "Время ответа — `pipeline.ask` целиком (поиск + генерация до 160 токенов), загрузка модели "
              "в замер не входит.", ""]
    L += ["## По вопросам", "",
          "Ранг — место первого фрагмента из документа-источника в гибридной выдаче (— = не в первых 10).", "",
          "| id | Вопрос | Эталон | Ранг hybrid | Покрытие | Отказ без LLM |" +
          (" Ответ LLM | Верно |" if ctx.get("llm") else ""),
          "|---|---|---|---|---|---|" + ("---|---|" if ctx.get("llm") else "")]
    for qid, q in qs.items():
        rank = r["hybrid"]["ranks"].get(qid)
        row = nl["rows"][qid]
        line = (f"| {qid} | {q['question']} | {q['answer'] or '(нет в корпусе)'} | "
                f"{'' if q['kind'] == 'trap' else (rank or '—')} | {row['coverage']} | "
                f"{'да' if row['refused'] else 'нет'} |")
        if ctx.get("llm"):
            lr = ctx["llm"]["rows"][qid]
            ans = lr["answer"].replace("|", "/").replace("\n", " ")[:120]
            line += f" {ans} | {'да' if lr['ok'] else 'нет'} |"
        L.append(line)
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "data" / "synthetic" / "docs"))
    ap.add_argument("--index", default=str(ROOT / "index_eval"))
    ap.add_argument("--llm", default="none", help="none | llamacpp | ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=None, help="Куда писать отчёт .md (по умолчанию eval/results_ДАТА.md)")
    ap.add_argument("--rescore", default=None, help="results_*.json: пересчитать метрики LLM по сохранённым ответам")
    ap.add_argument("--note", default="", help="Строка-примечание в начало отчёта")
    args = ap.parse_args()

    questions = load_jsonl(ROOT / "eval" / "questions.jsonl")
    if args.rescore:
        rescore(Path(args.rescore), questions, args.note)
        return
    calib_set = load_jsonl(ROOT / "eval" / "calibration.jsonl")
    pipe = RagPipeline(args.index, generator=make_generator(args.llm, args.model))
    print("Индексация…", flush=True)
    ingest = pipe.ingest(args.docs)
    info = pipe.info()
    print(json.dumps(ingest, ensure_ascii=False))

    calib = calibrate(pipe, calib_set)
    print(f"Калибровка: лучший порог {calib['best_threshold']} ({calib['correct']}/{calib['n']}); "
          f"в коде: {pipe.coverage_threshold}")
    if args.calibrate:
        print(json.dumps(calib, ensure_ascii=False, indent=1))
        return

    print("Поиск…", flush=True)
    retrieval = eval_retrieval(pipe, questions)
    for m in MODES:
        x = retrieval[m]
        print(f"  {m:7s} Recall@5={x['recall@5']:.2f} MRR@10={x['mrr@10']:.2f} "
              f"p50={x['p50_ms']:.1f} мс p95={x['p95_ms']:.1f} мс")
    nollm = eval_refusal_nollm(pipe, questions)
    print(f"Без LLM: отказы на ловушках {nollm['trap_refusal']:.2f}, ложные отказы {nollm['false_refusal']:.2f}")
    llm = None
    if args.llm != "none":
        print("Ответы LLM…", flush=True)
        llm = eval_llm(pipe, questions)
        print(f"LLM: точность {llm['answer_accuracy']:.2f}, отказы на ловушках {llm['trap_refusal']:.2f}, "
              f"ложные отказы {llm['false_refusal']:.2f}, p50 {llm['p50_ms']:.0f} мс, p95 {llm['p95_ms']:.0f} мс")

    date = dt.date.today().isoformat()
    ctx = {"date": date, "env": env_info(), "info": info, "ingest": ingest, "retrieval": retrieval,
           "nollm": nollm, "calib": calib, "llm": llm, "questions": questions, "note": args.note}
    out = Path(args.out) if args.out else ROOT / "eval" / f"results_{date}.md"
    write_report(out, ctx)
    (out.with_suffix(".json")).write_text(json.dumps({k: v for k, v in ctx.items() if k != "questions"},
                                                     ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    main()
