# TalentMatch AI - Constructor Local Windows (Sin bloqueo SmartScreen / Mark of the Web)
# Ejecución en PowerShell:  powershell -ExecutionPolicy Bypass -File .\setup_local_windows.ps1
$ErrorActionPreference = "Stop"
$dir = Join-Path (Get-Location) "TalentMatch_AI_Local"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Write-Host "Carpeta local creada en: $dir" -ForegroundColor Cyan
Write-Host "1. Copie index.html dentro de $dir" -ForegroundColor Green
Write-Host "2. Ejecute 'python -m http.server 8080' o abra index.html en Chrome/Edge." -ForegroundColor Green
