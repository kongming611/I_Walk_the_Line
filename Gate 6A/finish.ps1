param([int]$InferencePid = 0)
$ErrorActionPreference = 'Stop'
$python = 'D:\miniconda3\python.exe'
$receipt = Join-Path $PSScriptRoot 'results\inference_freeze.json'
if ($InferencePid -ne 0) {
    Write-Host 'Waiting for the already running local inference worker.'
    while (-not (Test-Path -LiteralPath $receipt)) {
        if (-not (Get-Process -Id $InferencePid -ErrorAction SilentlyContinue)) {
            throw 'Inference worker exited before its completion receipt. No API judgment or reference evaluation was started.'
        }
        Start-Sleep -Seconds 5
    }
} else {
    & $python -u (Join-Path $PSScriptRoot 'run.py') infer
    if ($LASTEXITCODE -ne 0) { throw 'Inference failed.' }
}
& $python -u (Join-Path $PSScriptRoot 'audit.py')
if ($LASTEXITCODE -ne 0) { throw 'Inference audit failed.' }
foreach ($stage in @('analyze', 'judge', 'evaluate')) {
    & $python -u (Join-Path $PSScriptRoot 'run.py') $stage
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $stage" }
}
& $python -u (Join-Path $PSScriptRoot 'audit.py') --evaluated
if ($LASTEXITCODE -ne 0) { throw 'Final audit failed.' }
& $python -u (Join-Path $PSScriptRoot 'deliver.py')
if ($LASTEXITCODE -ne 0) { throw 'Delivery failed.' }
Write-Host 'COMPLETED: results\REPORT.md and results\final_audit.json'
