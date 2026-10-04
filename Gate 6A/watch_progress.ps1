# Read-only observer. Closing this window does not affect either experiment worker.
$ErrorActionPreference = 'Continue'
$results = Join-Path $PSScriptRoot 'results'
$Host.UI.RawUI.WindowTitle = 'Sachs experiment - READ ONLY progress monitor'
Write-Host 'Sachs: Self-Compatibility vs DeepSeek + Jev' -ForegroundColor Cyan
Write-Host 'READ ONLY: watches saved outputs every 3 seconds; does not control the experiment.'
Write-Host 'This displays live file-based progress, not the original worker stdout.'
Write-Host 'You may close THIS window at any time without stopping the experiment.' -ForegroundColor Green
Write-Host ('Results: ' + $results)
Write-Host ''
$previous = ''
while ($true) {
    try {
        $files = @(Get-ChildItem -LiteralPath (Join-Path $results 'inference') -Filter '*.npz' -File -Recurse -ErrorAction SilentlyContinue)
        $latest = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $completedRuns = @(Get-ChildItem -LiteralPath (Join-Path $results 'inference') -Filter 'sc_size5.json' -File -Recurse -ErrorAction SilentlyContinue).Count
        $judgments = @(Get-ChildItem -LiteralPath (Join-Path $results 'judgments') -Filter '*.json' -File -ErrorAction SilentlyContinue).Count
        $stage = 'CDFM inference'
        if (Test-Path -LiteralPath (Join-Path $results 'inference_freeze.json')) { $stage = 'Inference finished; audit / Jev judgments' }
        if (Test-Path -LiteralPath (Join-Path $results 'prediction_freeze.json')) { $stage = 'Judgments frozen; evaluation / final audit' }
        if (Test-Path -LiteralPath (Join-Path $results 'delivery_receipt.json')) { $stage = 'COMPLETED' }
        $attempts = 0
        $reserved = 0.0
        $ledgerPath = Join-Path $results 'api_ledger.json'
        if (Test-Path -LiteralPath $ledgerPath) {
            $ledger = @(Get-Content -LiteralPath $ledgerPath -Raw -Encoding UTF8 | ConvertFrom-Json)
            $attempts = $ledger.Count
            foreach ($entry in $ledger) { $reserved += [double]$entry.budget_charge_usd }
        }
        $latestName = if ($latest) { $latest.Directory.Name + '/' + $latest.Name } else { '(first forward pass running)' }
        $state = "$stage|$($files.Count)|$completedRuns|$judgments|$attempts|$latestName"
        if ($state -ne $previous) {
            $now = Get-Date -Format 'HH:mm:ss'
            Write-Host "[$now] $stage" -ForegroundColor Cyan
            Write-Host ('  Forward caches: {0}/1701 ({1:P1}) | Finished data runs: {2}/21 | Jev decisions: {3}/21' -f $files.Count, ($files.Count / 1701.0), $completedRuns, $judgments)
            Write-Host ('  Latest: {0} | API attempts: {1} | Conservative budget used: ${2:F6} / $2' -f $latestName, $attempts, $reserved)
            if ($latest) { Write-Host ('  Latest cache saved: ' + $latest.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')) }
            Write-Host ''
            $previous = $state
        }
        if ($stage -eq 'COMPLETED') {
            Write-Host ('Final report: ' + (Join-Path $results 'REPORT.md')) -ForegroundColor Green
            break
        }
    } catch {
        Write-Host 'An output is being updated; the monitor will retry.' -ForegroundColor DarkGray
    }
    Start-Sleep -Seconds 3
}
