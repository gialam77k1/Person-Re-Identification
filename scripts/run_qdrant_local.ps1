param(
    [string]$EnvFile = ".env.qdrant",
    [string]$ComposeFile = "docker-compose.qdrant.yml",
    [switch]$Detach
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path $ComposeFile)) {
    throw "Compose file not found: $ComposeFile"
}

if (-not (Test-Path $EnvFile)) {
    throw "Env file not found: $EnvFile. Copy .env.qdrant.example to .env.qdrant first."
}

$arguments = @(
    "compose",
    "--env-file", $EnvFile,
    "-f", $ComposeFile,
    "up"
)

if ($Detach) {
    $arguments += "-d"
}

Write-Output "Starting local Qdrant:"
Write-Output "docker $($arguments -join ' ')"
docker @arguments
exit $LASTEXITCODE
