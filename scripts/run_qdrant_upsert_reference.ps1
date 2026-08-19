param(
    [string]$PythonExe = "C:\tmp\reid-mlops\python.exe",
    [string]$CollectionName = "reid_reference",
    [Parameter(Mandatory = $true)]
    [string]$EmbeddingsPath,
    [Parameter(Mandatory = $true)]
    [string]$PidsPath,
    [Parameter(Mandatory = $true)]
    [string]$CamidsPath,
    [string]$QdrantUrl = "http://localhost:6333",
    [string]$OutputRoot = "artifacts/qdrant/local"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$arguments = @(
    "src/qdrant_local.py",
    "--qdrant-url", $QdrantUrl,
    "--output-root", $OutputRoot,
    "upsert-reference",
    "--collection-name", $CollectionName,
    "--embeddings-path", $EmbeddingsPath,
    "--pids-path", $PidsPath,
    "--camids-path", $CamidsPath
)

Write-Output "Upserting reference embeddings into local Qdrant:"
Write-Output "$PythonExe $($arguments -join ' ')"
& $PythonExe @arguments
exit $LASTEXITCODE
