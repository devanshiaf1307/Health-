import nbconvert
exporter = nbconvert.exporters.ScriptExporter()
script, _ = exporter.from_filename(r"C:\Devansh College\New folder\Blood group detection\Fingerprint-Blood-Group-Detection-main\Algos\FingerprintProcessor.ipynb")

with open("converted_script.py", "w", encoding="utf-8") as f:
    f.write(script)