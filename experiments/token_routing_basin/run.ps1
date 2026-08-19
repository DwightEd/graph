param(
    [string]$Root = "",
    [string]$Out = "",
    [string]$Python = "",
    [string]$Device = "cpu",
    [Nullable[int]]$Limit = $null,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
if (-not $Root) {
    $Root = Join-Path $projectRoot "..\data\RAGTruth\llama31_8b"
}
if (-not $Out) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Out = Join-Path $projectRoot "experiments\token_routing_basin\outputs\run_$stamp"
}
if (-not $Python) {
    $candidates = @()
    if ($env:CONDA_PREFIX) {
        $candidates += Join-Path $env:CONDA_PREFIX "python.exe"
    }
    $candidates += Join-Path $projectRoot ".audit_envs\research\python.exe"
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $Python) {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
}
if ($null -ne $Limit -and $Limit -lt 1) {
    throw "Limit must be a positive integer"
}
foreach ($split in @("train", "test")) {
    $manifest = Join-Path $Root "$split\manifest.json"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw "Missing canonical manifest: $manifest"
    }
}

$previousLocation = Get-Location
try {
    Set-Location $projectRoot
    New-Item -ItemType Directory -Force -Path (Join-Path $Out "logs") | Out-Null
    $limitArguments = @()
    if ($null -ne $Limit) {
        $limitArguments = @("--limit", "$Limit")
    }

    if (-not $SkipTests) {
        $ErrorActionPreference = "Continue"
        & $Python -m pytest -q tests/test_token_routing_basin.py tests/test_token_routing_basin_cli.py
        $ErrorActionPreference = "Stop"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $ErrorActionPreference = "Continue"
    & $Python -m experiments.token_routing_basin.main fit `
        --train-split (Join-Path $Root "train") `
        --output (Join-Path $Out "reference.npz") `
        --device $Device @limitArguments 2>&1 |
        ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.Exception.Message
            } else { $_.ToString() }
        } |
        Tee-Object -FilePath (Join-Path $Out "logs\fit.log")
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not (Test-Path -LiteralPath (Join-Path $Out "reference.npz"))) {
        throw "fit completed without reference.npz"
    }

    $ErrorActionPreference = "Continue"
    & $Python -m experiments.token_routing_basin.main score `
        --split-root (Join-Path $Root "test") `
        --reference (Join-Path $Out "reference.npz") `
        --output (Join-Path $Out "test_scores.npz") `
        --device $Device @limitArguments 2>&1 |
        ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.Exception.Message
            } else { $_.ToString() }
        } |
        Tee-Object -FilePath (Join-Path $Out "logs\score.log")
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not (Test-Path -LiteralPath (Join-Path $Out "test_scores.npz"))) {
        throw "score completed without test_scores.npz"
    }

    $ErrorActionPreference = "Continue"
    & $Python -m experiments.token_routing_basin.main evaluate `
        --split-root (Join-Path $Root "test") `
        --scores (Join-Path $Out "test_scores.npz") `
        --output-dir (Join-Path $Out "evaluation") `
        --device cpu 2>&1 |
        ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.Exception.Message
            } else { $_.ToString() }
        } |
        Tee-Object -FilePath (Join-Path $Out "logs\evaluate.log")
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not (Test-Path -LiteralPath (Join-Path $Out "evaluation\report.json"))) {
        throw "evaluate completed without report.json"
    }
    Write-Host "Completed token routing basin run: $Out"
}
finally {
    Set-Location $previousLocation
}
