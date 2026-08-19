param(
    [string]$EnvFile = ".env.triton",
    [string]$ComposeFile = "docker-compose.triton.yml",
    [switch]$Detach
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path $ComposeFile)) {
    throw "Compose file not found: $ComposeFile"
}

if (-not (Test-Path $EnvFile)) {
    throw "Env file not found: $EnvFile. Copy .env.triton.example to .env.triton and adjust TRITON_MODEL_REPOSITORY."
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

Write-Output "Starting Triton local server:"
Write-Output "docker $($arguments -join ' ')"
docker @arguments
exit $LASTEXITCODE
