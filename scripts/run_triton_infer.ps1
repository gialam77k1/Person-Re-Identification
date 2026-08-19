param(
    [string]$PythonExe = "C:\tmp\reid-mlops\python.exe",
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,
    [string]$ServerUrl = "http://localhost:8000",
    [string]$ModelName = "reid_embedding",
    [string]$InputName = "images",
    [string]$OutputName = "embeddings",
    [int]$InputHeight = 224,
    [int]$InputWidth = 224,
    [string]$OutputRoot = "artifacts/inference/local-triton"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe"
}

$arguments = @(
    "src/triton_infer.py",
    "--image-path", $ImagePath,
    "--server-url", $ServerUrl,
    "--model-name", $ModelName,
    "--input-name", $InputName,
    "--output-name", $OutputName,
    "--input-height", "$InputHeight",
    "--input-width", "$InputWidth",
    "--output-root", $OutputRoot
)

Write-Output "Running Triton local inference:"
Write-Output "$PythonExe $($arguments -join ' ')"
& $PythonExe @arguments
exit $LASTEXITCODE
