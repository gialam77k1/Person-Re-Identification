param(
    [string]$EnvFile = ".env.qdrant",
    [string]$ComposeFile = "docker-compose.qdrant.yml"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path $ComposeFile)) {
    throw "Compose file not found: $ComposeFile"
}

if (-not (Test-Path $EnvFile)) {
    throw "Env file not found: $EnvFile"
}

$arguments = @(
    "compose",
    "--env-file", $EnvFile,
    "-f", $ComposeFile,
    "down"
)

Write-Output "Stopping local Qdrant:"
Write-Output "docker $($arguments -join ' ')"
docker @arguments
exit $LASTEXITCODE
