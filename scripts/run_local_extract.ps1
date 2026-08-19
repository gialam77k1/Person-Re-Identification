param(
    [string]$PythonExe = "C:\tmp\reid-mlops\python.exe",
    [string]$ConfigPath = "configs/dadnet.yaml",
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [Parameter(Mandatory = $true)]
    [string]$CheckpointPath,
    [string]$RunSlug = "",
    [string]$RunRoot = "",
    [string[]]$ExtraOverrides = @()
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe"
}

if (-not $RunSlug) {
    $datasetSlug = ($DatasetName.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $RunSlug = "$datasetSlug-dadnet-extract-$timestamp"
}

if (-not $RunRoot) {
    $datasetSlug = ($DatasetName.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
    $RunRoot = Join-Path $projectRoot ("artifacts\" + $datasetSlug + "\" + $RunSlug)
}

$arguments = @(
    "src/extract_reference.py",
    "--config", $ConfigPath,
    "--checkpoint", $CheckpointPath,
    "--set", "data.dataset.name=$DatasetName",
    "--set", "data.location.root=$DatasetRoot",
    "--set", "runtime.run_slug=$RunSlug",
    "--set", "artifacts.run_root=$RunRoot"
)

foreach ($override in $ExtraOverrides) {
    $arguments += @("--set", $override)
}

Write-Output "Running local extract:"
Write-Output "$PythonExe $($arguments -join ' ')"
& $PythonExe @arguments
exit $LASTEXITCODE
