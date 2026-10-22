# FB15k 默认使用原配置的 100 轮/1024 batch；可用命令行参数覆盖。
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
python -u (Join-Path $taskRoot 'main.py') --task FB15k --config (Join-Path $taskRoot 'transe_configFB15K.json') @args
if ($LASTEXITCODE -ne 0) { throw 'TransE training failed.' }
