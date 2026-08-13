param(
    [string]$PythonExe = "C:\tmp\reid-mlops\python.exe",
    [string]$ConfigPath = "configs/dadnet.yaml",
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$RunSlug = "",
    [string[]]$TrainOverrides = @(),
    [string[]]$EvaluateOverrides = @()
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe"
}

$datasetSlug = ($DatasetName.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
if (-not $RunSlug) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $RunSlug = "$datasetSlug-dadnet-local-$timestamp"
}

$runRoot = Join-Path $projectRoot ("artifacts\" + $datasetSlug + "\" + $RunSlug)

& (Join-Path $projectRoot "scripts\run_local_train.ps1") `
    -PythonExe $PythonExe `
    -ConfigPath $ConfigPath `
    -DatasetName $DatasetName `
    -DatasetRoot $DatasetRoot `
    -RunSlug $RunSlug `
    -ExtraOverrides $TrainOverrides
if ($LASTEXITCODE -ne 0) {
    throw "Local train step failed"
}

$checkpointPath = Join-Path $runRoot "checkpoints\best_model.pth"
if (-not (Test-Path $checkpointPath)) {
    throw "Missing best checkpoint at $checkpointPath"
}

& (Join-Path $projectRoot "scripts\run_local_evaluate.ps1") `
    -PythonExe $PythonExe `
    -ConfigPath $ConfigPath `
    -DatasetName $DatasetName `
    -DatasetRoot $DatasetRoot `
    -CheckpointPath $checkpointPath `
    -RunSlug $RunSlug `
    -RunRoot $runRoot `
    -ExtraOverrides $EvaluateOverrides
exit $LASTEXITCODE
