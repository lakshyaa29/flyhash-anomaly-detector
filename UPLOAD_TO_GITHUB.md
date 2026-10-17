# Upload this project to GitHub

1. Extract `flyhash-anomaly-detector-github.zip`.
2. Open the extracted `flyhash-anomaly-detector` folder. `README.md`, `src/`, `results/` and `report/` should be directly inside it.
3. On GitHub, create a new empty repository named `flyhash-anomaly-detector`. Leave the automatic README, license and gitignore options unchecked because this folder already includes its own files.
4. Choose **uploading an existing file** / **Add file > Upload files**. Upload the folder's contents, including `.github/`, `.gitignore` and `.gitattributes`. Do not upload only the ZIP or an extra enclosing folder.
5. Commit the upload. Open the README and the Actions tab to check the result.

If uploading folders or hidden files through the browser is inconvenient, use Git from PowerShell **inside the extracted project folder**:

```powershell
git init
git add .
git commit -m "Add reproducible anomaly detection baselines and report"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/flyhash-anomaly-detector.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Sign in when Git prompts you. These commands are instructions for you; the project has not been published automatically.

The package already excludes raw datasets, processed arrays, prediction archives and the local Python environment. Download and reproduce them with the commands in README.md. Keep the published tables, plots, screenshots and report included.

Siddhesh's actual encoder and its measured comparison are marked pending. Add the real encoder later using `docs/ENCODER_HANDOFF.md`. The presentation is excluded.
