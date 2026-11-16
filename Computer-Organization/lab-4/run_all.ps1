param([string]$VivadoBin = 'D:\Software\AMDDesignTools\2026.1\Vivado\bin')
$ErrorActionPreference = 'Stop'
$projectRoot = Join-Path $PSScriptRoot 'lab4-verilog'
$buildDir = Join-Path $projectRoot '.verification'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
$rtl = @(Get-ChildItem -LiteralPath "$projectRoot\lab4-verilog.srcs\sources_1\new" -Filter '*.v' | ForEach-Object FullName)
$benches = @(Get-ChildItem -LiteralPath "$projectRoot\lab4-verilog.srcs\sim_1\new" -Filter '*.v' | ForEach-Object FullName)
$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add('Vivado XSim 2026.1 - Lab 4 self-checking behavioral simulations')
$summary.Add('Clock period 10 ns; synchronous writes and asynchronous reads.')
Copy-Item -LiteralPath (Join-Path $projectRoot 'lab4-verilog.srcs\sources_1\new\imem.hex') -Destination $buildDir
Push-Location -LiteralPath $buildDir
try {
    & "$VivadoBin\xvlog.bat" @rtl @benches --log compile.log
    if ($LASTEXITCODE -ne 0) { throw 'Verilog compilation failed' }
    @('log_wave -r /', 'run all', 'quit') | Set-Content -LiteralPath run.tcl -Encoding ascii
    foreach ($bench in $benches) {
        $top = [System.IO.Path]::GetFileNameWithoutExtension($bench)
        & "$VivadoBin\xelab.bat" $top --debug all --snapshot $top --log "$top-elaborate.log"
        if ($LASTEXITCODE -ne 0) { throw "Elaboration failed: $top" }
        & "$VivadoBin\xsim.bat" $top --tclbatch run.tcl --wdb "$top.wdb" --log "$top-simulate.log"
        if ($LASTEXITCODE -ne 0) { throw "Simulation failed: $top" }
        $lines = Get-Content -LiteralPath "$top-simulate.log"
        $pass = @($lines | Where-Object { $_ -match "^PASS $top`:" })
        if ($pass.Count -ne 1 -or ($lines -match '^Fatal:')) { throw "Missing PASS or failure in $top" }
        $summary.Add($pass[0])
    }
    $summary.Add('ALL 5 TESTBENCHES PASSED')
    $summary | Set-Content -LiteralPath (Join-Path $projectRoot '..\output.txt') -Encoding utf8
    $summary | ForEach-Object { Write-Output $_ }
} finally {
    Pop-Location
}

