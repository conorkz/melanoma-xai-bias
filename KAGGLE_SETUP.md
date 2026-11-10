# Kaggle setup for the RQ1 benchmark

This guide takes you from zero to the first Stage 1 result. The runnable notebook is `notebooks/kaggle_benchmark.ipynb`.

## 1. Account and verification

1. Sign in at https://www.kaggle.com.
2. Open https://www.kaggle.com/settings and verify your phone number. Kaggle requires phone verification before it enables GPU accelerators and internet access.
3. Check the GPU quota in Settings. The free tier is about 30 GPU hours per week, with sessions up to 12 hours.

## 2. Create the notebook

1. Go to https://www.kaggle.com/code and click **New Notebook**.
2. Open **File -> Import Notebook** and upload `notebooks/kaggle_benchmark.ipynb`.
3. In the right sidebar, open **Settings**:
   - **Accelerator:** GPU P100, or GPU T4 x2.
   - **Internet:** On.
4. Optional but faster: click **Add Input**, search for `Skin Cancer MNIST: HAM10000` by `kmader`, and attach it. The notebook links the images from `/kaggle/input/skin-cancer-mnist-ham10000`. If you skip this, the notebook downloads the archive from Harvard Dataverse.

## 3. Run Stage 1

Run the notebook cells from top to bottom.

- The environment cell prints the GPU. If it says `CUDA available: False`, stop and enable the GPU in Settings.
- The clone cell pulls `https://github.com/conorkz/melanoma-xai-bias` and installs the package. The repository is public.
- The data cell verifies 10,015 images and 10,015 metadata rows.
- The Stage 1 cell runs 24 configurations. It takes about 3 to 4 GPU hours.
- The collect cell writes `results/rq1_stage1.md` and `.csv` and displays the table.

If the session resets, rerun the clone, data, and Stage 1 cells. `--skip-existing` skips completed runs, but only if the `outputs/runs` folder survived. Download or save the output before closing the session.

If you already ran Stage 1 in another Kaggle session and saved the output, download that `outputs/runs` folder into this repo before the Stage 1 cell. The completed runs are reused and only missing configurations are trained.

## 4. Review the Stage 1 screen

The Stage 1 table is descriptive. In this pilot, the planned focal loss plus class-balanced sampling did not beat the CE baseline on validation. The strongest validation configuration was CE with strong augmentation. Stage 2 is pre-declared and does not need a winner selection.

Use `val_sensitivity_mean` to describe the screen. Do not use the test columns for selection.

## 5. Run Stage 2

Run the next cells:

- The Stage 2 cell runs 27 configurations: 3 architectures x 3 conditions x 3 seeds at 224 px. It takes about 5 to 7 GPU hours.
- The conditions are CE baseline, focal2 + class-balanced, and CE + strong augmentation.
- The collect cell writes `results/rq1_stage2.md` and `.csv`.
- The figures cell picks the median seed of the best Stage 2 configuration and displays the training curves and ROC curve.
- The package cell writes `teacher_update.zip` with `results/` and `outputs/runs/`.

## 6. Download the results

1. Open the **Output** pane in the notebook editor.
2. Download `teacher_update.zip`.
3. On your Mac:

```bash
cd /path/to/melanoma-xai-bias
unzip ~/Downloads/teacher_update.zip -d .
cp RESULTS_MEMO_TEMPLATE.md RESULTS_MEMO.md
open RESULTS_MEMO.md
```

4. Paste both tables into the memo and fill in the interpretation.

## 7. Persistence and limits

- `/kaggle/working` is auto-saved up to 20 GB. The HAM10000 images, run outputs, and checkpoints all fit, but the archive is not small.
- A session can run up to 12 hours. Stage 1 plus Stage 2 usually fits. If the session ends, download the results and resume in a new session.
- `--skip-existing` only skips runs whose `outputs/runs/<run_name>/metrics.json` still exists.
- If you use **Save & Run All**, the notebook runs from top to bottom in a fresh session. It will rerun any missing runs. For a long interactive run, keep the tab open and do not rely on the 20-minute idle timeout.

## 8. Optional Kaggle CLI

The web UI is enough. The CLI is useful for uploading datasets or notebooks later.

```bash
uv pip install --python .venv/bin/python kaggle
.venv/bin/kaggle auth login
```

Or use an API token from https://www.kaggle.com/settings/api:

```bash
export KAGGLE_API_TOKEN=your_token_here
```

The CLI also supports `~/.kaggle/access_token` and the legacy `~/.kaggle/kaggle.json`. Never commit the token to the repository.

## 9. Troubleshooting

- **GPU option greyed out:** verify your phone number in Settings.
- **Internet option greyed out:** verify your phone number, then restart the session.
- **CUDA out of memory:** lower `training.batch_size` in `configs/pilot.yaml` or in the generated Stage 2 config, then rerun the failed runs.
- **Repo clone fails:** the repository is public at `https://github.com/conorkz/melanoma-xai-bias`. If it has been made private, either make it public for the run or upload the repo as a Kaggle dataset and adjust the clone cell.
- **Data download is slow:** attach the `kmader/skin-cancer-mnist-ham10000` dataset and rerun the data cell. It links the images instead of downloading.
- **No runs found after collect:** the run folders are not under `outputs/runs`, or the session reset. Check `find outputs/runs -maxdepth 2 -name metrics.json | wc -l`.
- **Pretrained weight warning:** the Hugging Face unauthenticated warning is harmless. Set `HF_TOKEN` in Kaggle secrets if it becomes a rate limit.
