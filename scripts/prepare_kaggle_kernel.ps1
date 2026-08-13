param(
    [string]$JobConfigPath = "kaggle_setup/my_kernel/job-config.json"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$jobConfigFullPath = Join-Path $projectRoot $JobConfigPath
$kernelRoot = Split-Path -Parent $jobConfigFullPath
$buildRoot = Join-Path $kernelRoot "build"
$srcBuild = Join-Path $buildRoot "src"
$configsBuild = Join-Path $buildRoot "configs"

if (-not (Test-Path $jobConfigFullPath)) {
    throw "Missing job config at $jobConfigFullPath"
}

if (Test-Path $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $srcBuild | Out-Null
New-Item -ItemType Directory -Path $configsBuild | Out-Null

Copy-Item -Path (Join-Path $projectRoot "src\*") -Destination $srcBuild -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot "configs\dadnet.yaml") -Destination $configsBuild
Copy-Item -LiteralPath (Join-Path $kernelRoot "run_reid_kaggle.py") -Destination $buildRoot
Copy-Item -LiteralPath $jobConfigFullPath -Destination (Join-Path $buildRoot "job-config.json")

Get-ChildItem -Path $srcBuild -Directory -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force

$jobConfig = Get-Content -Raw $jobConfigFullPath | ConvertFrom-Json
$metadataTemplatePath = Join-Path $kernelRoot "kernel-metadata.template.json"
$metadata = Get-Content -Raw $metadataTemplatePath | ConvertFrom-Json
$metadata.id = $jobConfig.kernel_id
$metadata.title = "Person ReID MLOps - $($jobConfig.dataset_name)"
$metadata.enable_gpu = [bool]$jobConfig.enable_gpu
$metadata.enable_internet = [bool]$jobConfig.enable_internet
$metadata.dataset_sources = @($jobConfig.dataset_sources)
$metadata.code_file = "run_reid_kaggle.py"

$metadata | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $buildRoot "kernel-metadata.json") -Encoding UTF8
Write-Output $buildRoot
