param([switch]$Recalculate, [int]$Port = 4180)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    & py -3.12 -c "import numpy,pandas,flask,sklearn,scipy,threadpoolctl"
    if ($LASTEXITCODE -ne 0) {
        & py -3.12 -m pip install -r requirements-quantdinger-demo.txt
        if ($LASTEXITCODE -ne 0) { throw 'Research dependency installation failed' }
    }
    if ($Recalculate -or -not (Test-Path -LiteralPath 'research/workbuddy-results.json')) {
        & py -3.12 -m research.workbuddy_run --profile research --end 2026-11-09
        if ($LASTEXITCODE -ne 0) { throw 'Backtest failed; the previous report is preserved' }
    }
    Write-Host "CodeXQuant: http://127.0.0.1:$Port"
    & py -3.12 -m research.workbuddy_run --serve --port $Port
} finally {
    Pop-Location
}
