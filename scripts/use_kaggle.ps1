param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$configDir = Join-Path $projectRoot ".local\.kaggle"
$kaggleExe = "C:\tmp\reid-mlops\Scripts\kaggle.exe"

if (-not (Test-Path (Join-Path $configDir "kaggle.json"))) {
    throw "Missing Kaggle credentials at $configDir\kaggle.json"
}

if (-not (Test-Path $kaggleExe)) {
    throw "Kaggle CLI not found at $kaggleExe"
}

$env:KAGGLE_CONFIG_DIR = $configDir
& $kaggleExe @ArgsList
exit $LASTEXITCODE
