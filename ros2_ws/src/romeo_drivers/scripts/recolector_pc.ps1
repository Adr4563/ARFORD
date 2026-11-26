<#
Recolector de recursos de la WORKSTATION (PC de inferencia) mientras corre
Romeo. Es la contraparte en la PC de recolector_sistema.sh de la Raspberry.

Escribe una fila cada ~1 s con el consumo del computador: CPU total, RAM, y
de la GPU NVIDIA (uso, VRAM total y por proceso, temperatura y potencia). En
la PC corren el LLM (Ollama), STT (NeMo FastConformer), SER (emotion2vec+),
TER (RoBERTuito) y TTS (Piper): parte usa GPU y parte CPU, por eso se miden
ambos. Se cruza luego con los turnos (~/romeo_medicion_turnos.csv de la Pi)
por la marca de tiempo EN MILISEGUNDOS.

POR QUE ESTE DISENO:
- nvidia-smi corre en MODO CONTINUO (-lms 1000): el proceso emite una linea
  por muestra sin volver a arrancar cada vez, que es mas preciso y barato que
  invocarlo en un bucle. De ahi sale la cadencia real de 1 s.
- La CPU/RAM del sistema se leen con contadores de rendimiento crudos
  (NO Get-Counter, que tarda ~1 s por llamada y estiraria el intervalo).
- La marca de tiempo es epoch en MILISEGUNDOS (UtcNow), igual que debe serlo
  en la Pi, para que el cruce turno-a-turno sea valido.

No toca ROS ni Romeo: solo lee contadores de Windows y nvidia-smi. Ctrl+C para
parar.

Uso en la PC:
    powershell -ExecutionPolicy Bypass -File recolector_pc.ps1

Variables de entorno:
    ROMEO_PC_CSV   destino (por defecto <script>\..\..\..\..\mediciones_pc\sistema_pc_<fecha>.csv)
    ROMEO_PC_MS    periodo de muestreo en ms (por defecto 1000)

Columnas del CSV:
    t_ms             epoch unix UTC en MILISEGUNDOS
    hora             hora local legible (para leerlo a ojo)
    cpu_total_pct    uso de CPU del sistema (%)
    ram_usada_mb     RAM en uso (MB)
    ram_total_mb     RAM total (MB)
    gpu_pct          uso de la GPU (%)
    vram_usada_mb    VRAM en uso total (MB)
    vram_total_mb    VRAM total (MB)
    vram_ollama_mb   VRAM del proceso de Ollama (LLM), si corre (MB)
    gpu_temp_c       temperatura de la GPU (C)
    gpu_watios       potencia de la GPU (W)
#>

$ErrorActionPreference = "Stop"

$periodoMs = [int]($env:ROMEO_PC_MS); if ($periodoMs -le 0) { $periodoMs = 1000 }

$fecha = Get-Date -Format "yyyyMMdd_HHmmss"
if ($env:ROMEO_PC_CSV) {
    $csv = $env:ROMEO_PC_CSV
} else {
    $base = Join-Path $PSScriptRoot "..\..\..\..\mediciones_pc"
    $csv  = Join-Path $base "sistema_pc_$fecha.csv"
}
$dir = Split-Path -Parent $csv
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

"t_ms,hora,cpu_total_pct,ram_usada_mb,ram_total_mb,gpu_pct,vram_usada_mb,vram_total_mb,vram_ollama_mb,gpu_temp_c,gpu_watios" |
    Out-File -FilePath $csv -Encoding utf8

Write-Host "[recolector-pc] escribiendo en $csv cada ${periodoMs}ms (Ctrl+C para parar)"

# --- CPU del sistema: contador crudo via P/Invoke-less WMI performance ---
# Se usa la clase RAW de rendimiento y se calcula el % entre dos lecturas, en
# vez de Get-Counter (que bloquea ~1 s). Base: PercentProcessorTime del _Total.
function Get-CpuRaw {
    $p = Get-CimInstance Win32_PerfRawData_PerfOS_Processor -Filter "Name='_Total'"
    return @{ t = $p.PercentProcessorTime; ts = $p.Timestamp_Sys100NS }
}
$prev = Get-CpuRaw

# --- GPU: nvidia-smi en modo continuo, leyendo su salida linea por linea ---
# timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "nvidia-smi"
$psi.Arguments = "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits -lms $periodoMs"
$psi.RedirectStandardOutput = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$smi = [System.Diagnostics.Process]::Start($psi)

# VRAM por proceso (Ollama) se consulta aparte, cada vuelta, con una llamada
# puntual (es barata y no la da el -lms de arriba).
function Get-VramOllama {
    try {
        $apps = nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null
        if (-not $apps) { return 0 }
        $total = 0
        foreach ($l in $apps) {
            $c = ($l -split ',') | ForEach-Object { $_.Trim() }
            $pid = [int]$c[0]
            $mem = [int]$c[1]
            try {
                $proc = Get-Process -Id $pid -ErrorAction Stop
                if ($proc.ProcessName -match 'ollama') { $total += $mem }
            } catch { }
        }
        return $total
    } catch { return 0 }
}

try {
    while (-not $smi.HasExited) {
        $linea = $smi.StandardOutput.ReadLine()
        if ($null -eq $linea -or $linea.Trim() -eq "") { continue }

        $tms  = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $hora = Get-Date -Format "HH:mm:ss.fff"

        # GPU (de la linea de nvidia-smi)
        $g = ($linea -split ',') | ForEach-Object { $_.Trim() }
        $gpu = $g[0]; $vramU = $g[1]; $vramT = $g[2]; $gtemp = $g[3]; $gpow = $g[4]

        # CPU % entre esta lectura y la anterior
        $cur = Get-CpuRaw
        $dt = $cur.ts - $prev.ts
        $db = $cur.t  - $prev.t
        if ($dt -gt 0) {
            # PercentProcessorTime es tiempo OCIOSO acumulado; CPU = 100 - idle%
            $cpu = [math]::Round(100 - (100 * $db / $dt), 1)
            if ($cpu -lt 0) { $cpu = 0 }
        } else { $cpu = 0 }
        $prev = $cur

        # RAM
        $os = Get-CimInstance Win32_OperatingSystem
        $ramUsada = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1024)
        $ramTotal = [math]::Round($os.TotalVisibleMemorySize / 1024)

        $vramOllama = Get-VramOllama

        "$tms,$hora,$cpu,$ramUsada,$ramTotal,$gpu,$vramU,$vramT,$vramOllama,$gtemp,$gpow" |
            Out-File -FilePath $csv -Append -Encoding utf8
    }
} finally {
    if ($smi -and -not $smi.HasExited) { $smi.Kill() }
}
