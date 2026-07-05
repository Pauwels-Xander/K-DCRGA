# Pack code + artifacts for fusion cloud run (run from repo root).
# Produces fusion_code.tar.gz and fusion_artifacts.tar.gz in repo root.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..
Write-Host "Packing from $(Get-Location)"

# Code (local branch — fusion commits may not be on remote yet)
tar -czf fusion_code.tar.gz `
  pyproject.toml README.md LICENSE `
  src configs experiments tests `
  --exclude="experiments/cloud/*.tar.gz"

# Artifacts (gitignored on laptop)
tar -czf fusion_artifacts.tar.gz `
  third_party/knowddi `
  data/processed `
  runs/bench/kdcrga_dedicom_v5/seed_1 `
  runs/bench/kdcrga_dedicom_v5/seed_2

Get-Item fusion_code.tar.gz, fusion_artifacts.tar.gz | Format-Table Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
