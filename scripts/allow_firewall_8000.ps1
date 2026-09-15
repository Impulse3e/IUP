# Allow inbound TCP 8000 so other devices on the LAN can reach the IUP server.
# ASCII-only so Windows PowerShell 5.1 parses it regardless of file encoding.
$ErrorActionPreference = "Stop"

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Administrator rights required. Requesting elevation..."
    $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $arg
    exit 0
}

$ruleName = "IUP Server 8000"
$existing = $null
try {
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
} catch {
    $existing = $null
}

if ($existing) {
    Write-Host "Firewall rule already exists: $ruleName"
} else {
    $created = $false
    try {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any | Out-Null
        $created = $true
    } catch {
        netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow protocol=TCP localport=8000 | Out-Null
        if ($LASTEXITCODE -eq 0) { $created = $true }
    }
    if (-not $created) {
        Write-Host "Could not add the firewall rule."
        exit 1
    }
    Write-Host "Added inbound TCP 8000 rule: $ruleName"
}

Write-Host ""
Write-Host "This PC IPv4 addresses (use one from the classroom LAN):"
ipconfig | findstr /R /C:"IPv4"
Write-Host ""
Write-Host "Teacher:  http://<this-ip>:8000"
Write-Host "Student:  same URL in IUP Student -> Show server address"
Write-Host "Done."
