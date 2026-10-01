# Windows PowerShell. Использование: .\make.ps1 <цель>
# Цели: install, synth, questions, ingest, serve, test, eval, eval-llm, clean
# Если PowerShell запрещает скрипты: powershell -ExecutionPolicy Bypass -File .\make.ps1 test
param(
    [Parameter(Position = 0)][string]$Target = "help",
    [string]$Model = "models\qwen2.5-3b-instruct-q4_k_m.gguf"
)

$ErrorActionPreference = "Stop"
$Py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

switch ($Target) {
    "install"   { & $Py -m pip install -r requirements.txt }
    "synth"     { & $Py data\synthetic\make_synthetic.py }
    "questions" { & $Py eval\make_questions.py }
    "ingest"    { & $Py -m local_doc_rag ingest data\synthetic\docs }
    "serve"     { & $Py -m local_doc_rag serve --no-llm }
    "test"      { & $Py -m pytest -q }
    "eval"      { & $Py eval\run_eval.py }
    "eval-llm"  { & $Py eval\run_eval.py --llm llamacpp --model $Model }
    "clean"     {
        foreach ($d in @("index", "index_eval", ".pytest_cache")) {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
    }
    default {
        Write-Host "Цели: install, synth, questions, ingest, serve, test, eval, eval-llm, clean"
    }
}
if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
