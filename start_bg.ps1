$env:HF_ENDPOINT = "https://hf-mirror.com"
$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = "C:\Users\阿龙\Desktop\medical_rag\venv\Scripts\pythonw.exe"
$pinfo.Arguments = "-u C:\Users\阿龙\Desktop\medical_rag\start_app.py"
$pinfo.WorkingDirectory = "C:\Users\阿龙\Desktop\medical_rag"
$pinfo.UseShellExecute = $false
$pinfo.CreateNoWindow = $true
$p = [System.Diagnostics.Process]::Start($pinfo)
Write-Output "PID: $($p.Id)"
