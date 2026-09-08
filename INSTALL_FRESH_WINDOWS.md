# TFM Neurotechnology Platform v1.0 — Fresh Windows installation

## 1. Install Miniforge

Install **Miniforge3 for Windows x86_64** using its normal graphical installer.

After installation, open **Miniforge Prompt** from the Windows Start menu.

Do not use a normal Command Prompt for the first setup unless `conda` is already
available there.

## 2. Extract the project

Extract the final project ZIP to a short, writable path, for example:

```text
C:\Users\<YOUR_USER>\Documents\TFM_Neurotechnology_Platform_v1.0
```

Avoid OneDrive-synchronised folders for the first smoke test.

## 3. Add the environment files

Copy these files into the project root:

```text
environment_windows.yml
environment_macos.yml
verify_environment.py
```

Keep the original `environment-lock.yml` as the exact Windows-development
snapshot used during thesis development.

## 4. Create the Windows environment

In **Miniforge Prompt**:

```bat
cd /d C:\Users\<YOUR_USER>\Documents\TFM_Neurotechnology_Platform_v1.0
conda env create -f environment_windows.yml
```

This can take several minutes because PsychoPy, PyTorch and wxPython are large.

If an old partial environment already exists:

```bat
conda env remove -n tfm_unified
conda env create -f environment_windows.yml
```

## 5. Activate it

```bat
conda activate tfm_unified
```

## 6. Verify the environment

```bat
python verify_environment.py
```

Expected final line:

```text
Environment verification PASSED.
```

Also verify that the exported decoder can be loaded:

```bat
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

CPU-only execution is acceptable for this platform.

## 7. Compile-check the PsychoPy experiments

```bat
python -m py_compile psychopy\Experiment_Offline.py
python -m py_compile psychopy\Experiment_Online_Silent.py
python -m py_compile psychopy\Experiment_Online_Imagined.py
```

No output means success.

## 8. Run the project readiness check

```bat
python check_export_readiness.py
```

Read any warnings carefully. Do not treat missing acquisition hardware as a
packaging failure during a software-only smoke test.

## 9. Launch the GUI

Preferred:

```bat
launch_gui.bat
```

Fallback:

```bat
python code\tfm_GUI\GUI.py
```

## 10. Windows smoke test

Confirm all of the following:

1. The GUI opens.
2. `Prepare Run` creates `data\latest_prepared_run.json`.
3. The selected PsychoPy experiment launches.
4. The bundled `LabRecorder\LabRecorder.exe` starts automatically.
5. LabRecorder opens without a missing-DLL error.
6. A short test creates an `.xdf` file under `recordings`.
7. The selected decoder bundle loads without an exception.
8. The experiment and GUI close cleanly.

## 11. When the test passes

Create the final archive as:

```text
TFM_Neurotechnology_Platform_v1.0.zip
```

Include:

```text
environment_windows.yml
environment_macos.yml
environment-lock.yml
verify_environment.py
```

The curated Windows file is the installation entry point. The lock file is the
historical exact Windows environment snapshot.
