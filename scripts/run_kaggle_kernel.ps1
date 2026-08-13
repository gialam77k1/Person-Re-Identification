param(
    [string]$JobConfigPath = "kaggle_setup/my_kernel/job-config.json",
    [int]$PollSeconds = 30
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$jobConfigFullPath = Join-Path $projectRoot $JobConfigPath
if (-not (Test-Path $jobConfigFullPath)) {
    throw "Missing job config at $jobConfigFullPath"
}

$jobConfig = Get-Content -Raw $jobConfigFullPath | ConvertFrom-Json
$kernelId = [string]$jobConfig.kernel_id
$downloadDir = Join-Path $projectRoot ([string]$jobConfig.download_dir)

$buildRoot = & (Join-Path $projectRoot "scripts\prepare_kaggle_kernel.ps1") -JobConfigPath $JobConfigPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to prepare Kaggle kernel bundle"
}

& (Join-Path $projectRoot "scripts\use_kaggle.ps1") kernels push -p $buildRoot
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle kernel push failed"
}

while ($true) {
    $statusOutput = & (Join-Path $projectRoot "scripts\use_kaggle.ps1") kernels status $kernelId 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to fetch Kaggle kernel status for $kernelId`n$statusOutput"
    }

    $statusText = ($statusOutput | Out-String).Trim()
    Write-Output $statusText
    $statusLower = $statusText.ToLowerInvariant()

    if ($statusLower -match "complete" -or $statusLower -match "successful") {
        break
    }
    if ($statusLower -match "error" -or $statusLower -match "failed" -or $statusLower -match "cancelled") {
        throw "Kaggle kernel run did not complete successfully: $statusText"
    }

    Start-Sleep -Seconds $PollSeconds
}

New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
& (Join-Path $projectRoot "scripts\use_kaggle.ps1") kernels output $kernelId -p $downloadDir -o
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download Kaggle kernel output"
}

Write-Output "Downloaded output to $downloadDir"
