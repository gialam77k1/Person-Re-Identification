param(
    [string]$PythonExe = "C:\tmp\reid-mlops\python.exe",
    [string]$ConfigPath = "configs/dadnet.yaml",
    [string]$ModelRoot = "model",
    [string]$CheckpointPath = "",
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$RunSlug = "",
    [string[]]$EvaluateOverrides = @(),
    [string[]]$ExtractOverrides = @()
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe"
}

$resolvedModelRoot = Resolve-Path $ModelRoot -ErrorAction Stop
if (-not $CheckpointPath) {
    $CheckpointPath = Join-Path $resolvedModelRoot "checkpoints\best_model.pth"
}

if (-not (Test-Path $CheckpointPath)) {
    throw "Checkpoint not found at $CheckpointPath"
}

$datasetSlug = ($DatasetName.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
if (-not $RunSlug) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $RunSlug = "$datasetSlug-dadnet-imported-$timestamp"
}

$runRoot = Join-Path $projectRoot ("artifacts\" + $datasetSlug + "\" + $RunSlug)

Write-Output "Using imported model root: $resolvedModelRoot"
Write-Output "Using checkpoint: $CheckpointPath"
Write-Output "Output run root: $runRoot"

& (Join-Path $projectRoot "scripts\run_local_evaluate.ps1") `
    -PythonExe $PythonExe `
    -ConfigPath $ConfigPath `
    -DatasetName $DatasetName `
    -DatasetRoot $DatasetRoot `
    -CheckpointPath $CheckpointPath `
    -RunSlug $RunSlug `
    -RunRoot $runRoot `
    -ExtraOverrides $EvaluateOverrides
if ($LASTEXITCODE -ne 0) {
    throw "Imported model evaluate step failed"
}

& (Join-Path $projectRoot "scripts\run_local_extract.ps1") `
    -PythonExe $PythonExe `
    -ConfigPath $ConfigPath `
    -DatasetName $DatasetName `
    -DatasetRoot $DatasetRoot `
    -CheckpointPath $CheckpointPath `
    -RunSlug $RunSlug `
    -RunRoot $runRoot `
    -ExtraOverrides $ExtractOverrides
if ($LASTEXITCODE -ne 0) {
    throw "Imported model extract step failed"
}
