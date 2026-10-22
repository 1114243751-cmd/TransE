# 从任意终端目录运行；其余参数直接传给 main.py，例如 --epochs 2 --model-dir ...。
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
python -u (Join-Path $taskRoot 'main.py') --config (Join-Path $taskRoot 'transe_configtiny.json') --do-eval @args
if ($LASTEXITCODE -ne 0) { throw 'TransE training failed.' }
