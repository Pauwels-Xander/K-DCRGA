# Pack tarballs for 4-box parallel fusion run (from repo root).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..
Write-Host "Packing from $(Get-Location)"

# Shared code (all boxes need experiments/fusion script path)
tar -czf fusion_code.tar.gz pyproject.toml README.md LICENSE src configs experiments tests

# K-DCRGA box: processed data + reused seed 1/2 checkpoints + knowddi eval txts
tar -czf fusion_kdcrga_data.tar.gz data/processed `
  runs/bench/kdcrga_dedicom_v5/seed_1 `
  runs/bench/kdcrga_dedicom_v5/seed_2 `
  third_party/knowddi/data

# KnowDDI boxes: full vendored tree (includes BioSNAP LMDB data)
tar -czf fusion_knowddi_data.tar.gz third_party/knowddi

Get-Item fusion_code.tar.gz, fusion_kdcrga_data.tar.gz, fusion_knowddi_data.tar.gz |
  Format-Table Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
