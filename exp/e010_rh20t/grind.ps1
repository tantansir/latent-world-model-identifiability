# Self-healing training grind, fired by Windows Task Scheduler every 10 min.
# Exits immediately if done or if another instance holds the lock.
$dir = "C:\Users\Kaizh\Desktop\physical representation\exp\e010_rh20t"
Set-Location $dir
$lock = Join-Path $dir "grind.lock"
if ((Test-Path "results\VFX_s0_c1.json") -and (Test-Path "results\V_s0_c1.json")) { exit }
if (Test-Path $lock) {
    $age = (Get-Date) - (Get-Item $lock).LastWriteTime
    if ($age.TotalMinutes -lt 9) { exit }   # another instance likely alive
}
New-Item -ItemType File -Force $lock | Out-Null
$env:E010_DATA = "data_cfg1"
try {
    if (-not (Test-Path "results\VFX_s0_c1.json")) {
        & python train.py --variant VFX --seed 0 *>> grind.log
    } elseif (-not (Test-Path "results\V_s0_c1.json")) {
        & python train.py --variant V --seed 0 *>> grind.log
    }
} finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}
