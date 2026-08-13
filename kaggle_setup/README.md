# Kaggle Automation

## 1. Credentials

Each teammate keeps their own Kaggle token in:

```text
.local/.kaggle/kaggle.json
```

This path is ignored by git.

## 2. Create your local job config

Copy the template:

```powershell
Copy-Item .\kaggle_setup\my_kernel\job-config.template.json .\kaggle_setup\my_kernel\job-config.json
```

Then edit:

- `kernel_id`: your Kaggle kernel slug, e.g. `yourname/person-reid-mlops`
- `dataset_name`: `market1501`, `dukemtmc-reid`, or `msmt17`
- `kaggle_dataset_root`: dataset path inside Kaggle, e.g. `/kaggle/input/...`
- `dataset_sources`: Kaggle dataset slugs attached to the kernel
- `overrides`: extra `--set` overrides if needed

## 3. Run end-to-end

```powershell
.\scripts\run_kaggle_kernel.ps1
```

This will:

1. prepare a Kaggle kernel bundle from the current repo
2. push the kernel to Kaggle
3. poll until the run finishes
4. download output files to `kaggle_outputs/`

## 4. Useful commands

Check CLI:

```powershell
.\scripts\use_kaggle.ps1 --version
```

Prepare bundle only:

```powershell
.\scripts\prepare_kaggle_kernel.ps1
```
