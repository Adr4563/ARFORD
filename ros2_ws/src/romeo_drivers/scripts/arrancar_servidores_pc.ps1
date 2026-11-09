# Arranca en la PC los servidores que usa Romeo: STT (8082) y TTS (8083).
#
# La voz la elige el parametro -Voz:
#   .\arrancar_servidores_pc.ps1              -> voz mexicana (vits, por defecto)
#   .\arrancar_servidores_pc.ps1 -Voz infantil  -> voz Supertonic M5 (infantil)
#
# El chat (Ollama / gemma3:4b) se arranca aparte (ollama ya corre como servicio).
param(
    [ValidateSet("mexicana", "vits", "infantil", "supertonic")]
    [string]$Voz = "mexicana"
)

$ErrorActionPreference = "Stop"
$scripts = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $scripts "..\..\..\..")).Path

# Normalizar el nombre de la voz al motor real.
$motor = if ($Voz -in "infantil", "supertonic") { "supertonic" } else { "vits" }

Write-Host "=== Servidores de Romeo en la PC ===" -ForegroundColor Cyan
Write-Host "Voz elegida: $Voz  ->  motor TTS: $motor"

# --- Liberar puertos si hay servidores viejos ---
foreach ($puerto in 8082, 8083) {
    $c = Get-NetTCPConnection -State Listen -LocalPort $puerto -ErrorAction SilentlyContinue
    if ($c) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "  puerto $puerto liberado (PID $($c.OwningProcess))"
    }
}
Start-Sleep -Seconds 1

# --- STT (reconocimiento de voz + emocion). Siempre vits-independiente. ---
Start-Process python -ArgumentList "stt_server.py" -WorkingDirectory $scripts -WindowStyle Minimized
Write-Host "  STT arrancando en 8082..."

# --- TTS (voz). El motor va por variable de entorno. ---
$env:ROMEO_TTS_MOTOR = $motor
if ($motor -eq "supertonic") {
    $env:ROMEO_TTS_SUPERTONIC_DIR = Join-Path $repo "modelos\voz-supertonic-m5"
}
Start-Process python -ArgumentList "tts_server.py" -WorkingDirectory $scripts -WindowStyle Minimized
Write-Host "  TTS arrancando en 8083 (motor $motor)..."

# --- Esperar a que ambos respondan ---
Write-Host "Esperando a que carguen los modelos..."
foreach ($s in @(@{n="STT"; p=8082}, @{n="TTS"; p=8083})) {
    $ok = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 2
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$($s.p)/salud" -TimeoutSec 3 -UseBasicParsing
            if ($r.Content -match '"listo":\s*true') { Write-Host "  $($s.n) LISTO: $($r.Content)" -ForegroundColor Green; $ok = $true; break }
        } catch {}
    }
    if (-not $ok) { Write-Host "  $($s.n) no respondio (revisa la ventana minimizada)" -ForegroundColor Yellow }
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -eq 'Wi-Fi' }).IPAddress
Write-Host ""
Write-Host "IP de esta PC (la que va en el launcher de la Pi): $ip" -ForegroundColor Cyan
