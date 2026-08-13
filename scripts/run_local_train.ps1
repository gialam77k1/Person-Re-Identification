param(
    [string]$PythonExe = "C:\tmp\reid-mlops\python.exe",
    [string]$ConfigPath = "configs/dadnet.yaml",
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$RunSlug = "",
    [string[]]$ExtraOverrides = @()
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe"
}

$datasetSlug = ($DatasetName.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
if (-not $RunSlug) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $RunSlug = "$datasetSlug-dadnet-train-$timestamp"
}

$runRoot = Join-Path $projectRoot ("artifacts\" + $datasetSlug + "\" + $RunSlug)

$arguments = @(
    "src/train.py",
    "--config", $ConfigPath,
    "--set", "data.dataset.name=$DatasetName",
    "--set", "data.location.root=$DatasetRoot",
    "--set", "runtime.run_slug=$RunSlug",
    "--set", "artifacts.run_root=$runRoot"
)

foreach ($override in $ExtraOverrides) {
    $arguments += @("--set", $override)
}

Write-Output "Running local train:"
Write-Output "$PythonExe $($arguments -join ' ')"
& $PythonExe @arguments
exit $LASTEXITCODE
