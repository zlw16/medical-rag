Dim shell
Set shell = CreateObject("WScript.Shell")
shell.Run "cmd.exe /c cd /d C:\Users\阿龙\Desktop\medical_rag && venv\Scripts\pythonw.exe -u start_app.py", 0, False
Set shell = Nothing
