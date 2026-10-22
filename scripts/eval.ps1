# 第一个参数为模型路径，其余参数原样传递，例如 --filtered。
$ErrorActionPreference = 'Stop'
if ($args.Count -lt 1) { throw 'Usage: ./scripts/eval.ps1 MODEL_PATH [--filtered] [--model-dir OUTPUT_DIR]' }
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskModelPath = $args[0]
$taskExtraArgs = @($args | Select-Object -Skip 1)
python -u (Join-Path $taskRoot 'evaluate.py') --eval-model-path $taskModelPath @taskExtraArgs
if ($LASTEXITCODE -ne 0) { throw 'TransE evaluation failed.' }
