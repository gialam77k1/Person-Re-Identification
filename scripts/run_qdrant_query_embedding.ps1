param(
    [string]$PythonExe = "C:\tmp\reid-mlops\python.exe",
    [string]$CollectionName = "reid_reference",
    [Parameter(Mandatory = $true)]
    [string]$EmbeddingPath,
    [int]$Limit = 5,
    [string]$QdrantUrl = "http://localhost:6333",
    [string]$OutputRoot = "artifacts/qdrant/local"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$arguments = @(
    "src/qdrant_local.py",
    "--qdrant-url", $QdrantUrl,
    "--output-root", $OutputRoot,
    "query-embedding",
    "--collection-name", $CollectionName,
    "--embedding-path", $EmbeddingPath,
    "--limit", "$Limit"
)

Write-Output "Querying local Qdrant with embedding:"
Write-Output "$PythonExe $($arguments -join ' ')"
& $PythonExe @arguments
exit $LASTEXITCODE
