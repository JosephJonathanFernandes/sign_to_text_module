<#
.SYNOPSIS
  Run K-Fold GNN full fine-tuning (PowerShell helper)

.EXAMPLE
  .\scripts\run_kfold_gnn.ps1 -Folds 5 -Epochs 8 -Lr 0.0001
#>
param(
  [int]$Folds = 5,
  [int]$Epochs = 8,
  [double]$Lr = 0.0001,
  [string]$OutDir = "logs\$(Get-Date -Format 'yyyyMMdd_HHmmss')_kfold"
)

Write-Output "Run settings: folds=$Folds, epochs=$Epochs, lr=$Lr"
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
"Started at: $(Get-Date -Format o)" | Out-File -FilePath (Join-Path $OutDir 'run.log') -Encoding utf8

# Activate virtualenv if present
$venvActivate = Join-Path $PSScriptRoot '..\venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
  Write-Output "Activating venv: $venvActivate"
  & $venvActivate
} else {
  Write-Warning "No virtualenv activate script found at $venvActivate. Ensure dependencies are installed." 
}

pip install -r requirements.txt

$trainCmd = "python ..\train.py --kfold $Folds --epochs $Epochs --lr $Lr"
Write-Output "Running: $trainCmd"
& cmd /c $trainCmd 2>&1 | Tee-Object -FilePath (Join-Path $OutDir 'train.log')

"Finished at: $(Get-Date -Format o)" | Out-File -FilePath (Join-Path $OutDir 'run.log') -Append -Encoding utf8
