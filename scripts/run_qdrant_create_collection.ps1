param(
    [string]$PythonExe = "C:\tmp\reid-mlops\python.exe",
    [string]$CollectionName = "reid_reference",
    [int]$VectorSize = 512,
    [string]$Distance = "Cosine",
    [string]$QdrantUrl = "http://localhost:6333",
    [string]$OutputRoot = "artifacts/qdrant/local"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$arguments = @(
    "src/qdrant_local.py",
    "--qdrant-url", $QdrantUrl,
    "--output-root", $OutputRoot,
    "create-collection",
    "--collection-name", $CollectionName,
    "--vector-size", "$VectorSize",
    "--distance", $Distance
)

Write-Output "Creating local Qdrant collection:"
Write-Output "$PythonExe $($arguments -join ' ')"
& $PythonExe @arguments
exit $LASTEXITCODE
