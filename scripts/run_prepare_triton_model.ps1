param(
    [string]$PythonExe = "C:\tmp\reid-mlops\python.exe",
    [Parameter(Mandatory = $true)]
    [string]$OnnxPath,
    [string]$ModelName = "reid_embedding",
    [string]$ModelVersion = "1",
    [int]$MaxBatchSize = 16,
    [int]$InputHeight = 224,
    [int]$InputWidth = 224,
    [int]$EmbeddingDim = 512,
    [string]$PreferredBatchSizes = "4,8,16",
    [string]$OutputRoot = ""
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe"
}

if (-not $OutputRoot) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputRoot = Join-Path $projectRoot ("artifacts\triton\triton-repository-" + $timestamp + "\model_repository")
}

$arguments = @(
    "src/prepare_triton_model.py",
    "--onnx-path", $OnnxPath,
    "--output-root", $OutputRoot,
    "--model-name", $ModelName,
    "--model-version", $ModelVersion,
    "--max-batch-size", "$MaxBatchSize",
    "--input-height", "$InputHeight",
    "--input-width", "$InputWidth",
    "--embedding-dim", "$EmbeddingDim",
    "--preferred-batch-sizes", $PreferredBatchSizes
)

Write-Output "Preparing Triton model repository:"
Write-Output "$PythonExe $($arguments -join ' ')"
& $PythonExe @arguments
exit $LASTEXITCODE
