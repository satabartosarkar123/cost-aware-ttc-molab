# Setup Script for Host Computer
# This script retrieves the necessary large files that are not tracked in git due to size limits.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$targetDir = Join-Path $scriptDir "ttc-frugalreason-poc\experiment_fr"

Write-Host "Navigating to experiment directory: $targetDir"
Set-Location $targetDir

# 1. Download OllamaSetup.exe
$ollamaInstaller = "OllamaSetup.exe"
if (-Not (Test-Path $ollamaInstaller)) {
    Write-Host "Downloading OllamaSetup.exe..."
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $ollamaInstaller
    Write-Host "OllamaSetup.exe downloaded successfully."
} else {
    Write-Host "OllamaSetup.exe already exists, skipping download."
}

# 2. Clone the PRM800K dataset repository
$prmDir = "temp_prm800k"
if (-Not (Test-Path $prmDir)) {
    Write-Host "Cloning PRM800K dataset repository..."
    git clone https://github.com/openai/prm800k.git $prmDir
    Write-Host "PRM800K cloned successfully."
} else {
    Write-Host "Directory $prmDir already exists, skipping clone."
}

Write-Host "============================================="
Write-Host "Setup complete!"
Write-Host "To run Ollama on this host, execute: .\$ollamaInstaller"
Write-Host "============================================="
