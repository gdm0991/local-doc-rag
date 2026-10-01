# Linux/macOS. На Windows — make.ps1 с теми же целями.
PY ?= python3
MODEL ?= models/qwen2.5-3b-instruct-q4_k_m.gguf

.PHONY: install synth questions ingest serve test eval eval-llm clean

install:
	$(PY) -m pip install -r requirements.txt

synth:
	$(PY) data/synthetic/make_synthetic.py

questions:
	$(PY) eval/make_questions.py

ingest:
	$(PY) -m local_doc_rag ingest data/synthetic/docs

serve:
	$(PY) -m local_doc_rag serve --no-llm

test:
	$(PY) -m pytest -q

eval:
	$(PY) eval/run_eval.py

eval-llm:
	$(PY) eval/run_eval.py --llm llamacpp --model $(MODEL)

clean:
	rm -rf index index_eval .pytest_cache
