$ErrorActionPreference = 'Stop'
$lassoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logPath = Join-Path $lassoDir 'output\gaussian-table-run.log'
$matlabPath = 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe'

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class PowerState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@

$ES_CONTINUOUS = [Convert]::ToUInt32('80000000',16)
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$ES_DISPLAY_REQUIRED = [uint32]0x00000002
$keepAwake = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED
[PowerState]::SetThreadExecutionState($keepAwake) | Out-Null

try {
    Push-Location $lassoDir
    try {
        & $matlabPath '-batch' "run('run_gaussian_table.m')" *> $logPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "MATLAB exited with code $exitCode. See $logPath"
    }
}
finally {
    [PowerState]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}

Start-Sleep -Seconds 10
rundll32.exe powrprof.dll,SetSuspendState 0,1,0
