param([int]$Limit=0)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$experimentDir=$PSScriptRoot
$manifest=Get-Content -Raw -LiteralPath (Join-Path $experimentDir 'split_manifest.json') | ConvertFrom-Json
$labels=@($manifest.P.trajectories | Where-Object split -eq 'train' | ForEach-Object label)
if($Limit -gt 0){$labels=@($labels | Select-Object -First $Limit)}
$downloadDir=Join-Path $experimentDir 'raw_training'
New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
$bridge=Join-Path (Split-Path $experimentDir) 'round3_20260921\remote.py'
$labels | ForEach-Object -Parallel {
 $ProgressPreference='SilentlyContinue';$ErrorActionPreference='Stop'
 $label=$_;$target=Join-Path $using:downloadDir "${label}_uvp_pointData.npz"
 $marker="$target.uploaded"
 if(Test-Path -LiteralPath $marker){return}
 $valid=$false
 for($attempt=1;$attempt -le 4;$attempt++){
  if(Test-Path -LiteralPath $target){
   & python -c 'import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None' $target 2>$null
   if($LASTEXITCODE -eq 0){$valid=$true;break}
  }
  try{
   $uri="https://huggingface.co/datasets/panxy1019/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/resolve/main/${label}_uvp_pointData.npz"
   if(Test-Path -LiteralPath $target){Invoke-WebRequest -Uri $uri -OutFile $target -Resume -TimeoutSec 100}
   else{Invoke-WebRequest -Uri $uri -OutFile $target -TimeoutSec 100}
  }catch{Write-Output "RETRY $label attempt=$attempt error=$($_.Exception.Message)"}
 }
 & python -c 'import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None' $target 2>$null
 if($LASTEXITCODE -ne 0){Write-Output "FAILED $label";return}
 & python $using:bridge put $target "/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/external_rnn_fno_20260923/raw_training/${label}_uvp_pointData.npz"
 if($LASTEXITCODE -eq 0){New-Item -ItemType File -Path $marker -Force | Out-Null;Write-Output "VERIFIED_UPLOADED $label"}
} -ThrottleLimit 4
