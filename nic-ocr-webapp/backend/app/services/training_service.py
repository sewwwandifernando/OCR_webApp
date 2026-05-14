import glob
import os
import queue
import shutil
import subprocess
import threading
from datetime import datetime
from typing import Generator

from sqlalchemy import text

_is_training: bool = False
_active_run_id: str | None = None
_log_queue: queue.Queue = queue.Queue()


def is_training() -> bool:
    return _is_training


def get_active_run_id() -> str | None:
    return _active_run_id


def get_log_lines() -> Generator[str, None, None]:
    while True:
        try:
            line = _log_queue.get(timeout=0.1)
            yield line
            if line == "[DONE]":
                break
        except queue.Empty:
            continue


def start_training(
    run_id: str,
    iterations: int,
    lstmf_list_file: str,
    settings,
    db_engine,
) -> None:
    global _is_training, _active_run_id
    _is_training = True
    _active_run_id = run_id

    thread = threading.Thread(
        target=_training_worker,
        args=(run_id, iterations, lstmf_list_file, settings, db_engine),
        daemon=True,
    )
    thread.start()


def _training_worker(
    run_id: str,
    iterations: int,
    lstmf_list_file: str,
    settings,
    db_engine,
) -> None:
    global _is_training, _active_run_id

    tessdata_prefix = settings.tessdata_prefix
    tessdata_best_prefix = "/usr/local/share/tessdata_best"  # best model for training
    tesseract_path = settings.tesseract_path
    base_sin_model = settings.base_sin_model
    storage_path = os.path.abspath(settings.storage_path)

    tesseract_dir = os.path.dirname(tesseract_path)
    lstmtraining_exe = "lstmtraining.exe" if os.name == "nt" else "lstmtraining"
    lstmtraining_path = os.path.join(tesseract_dir, lstmtraining_exe)

    run_dir = os.path.join(storage_path, "models", "history", run_id)
    os.makedirs(run_dir, exist_ok=True)
    output_base = os.path.join(run_dir, "sin_id")

    # Use tessdata_best for the LSTM source (float model, required for fine-tuning)
    lstm_source = os.path.join(tessdata_best_prefix, f"{base_sin_model}.lstm")

    # Count lstmf files for file_count
    try:
        with open(lstmf_list_file, "r", encoding="utf-8") as f:
            file_count = sum(1 for line in f if line.strip())
    except OSError:
        file_count = 0

    log_lines: list[str] = []
    success = False
    final_model_path: str | None = None

    # Extract .lstm from tessdata_best .traineddata if it doesn't exist yet
    if not os.path.exists(lstm_source):
        traineddata_for_extract = os.path.join(tessdata_best_prefix, f"{base_sin_model}.traineddata")
        combine_tessdata_exe = "combine_tessdata.exe" if os.name == "nt" else "combine_tessdata"
        combine_tessdata_path = os.path.join(tesseract_dir, combine_tessdata_exe)
        extract_result = subprocess.run(
            [combine_tessdata_path, "-e", traineddata_for_extract, lstm_source],
            capture_output=True,
            text=True,
        )
        extract_out = (extract_result.stdout + extract_result.stderr).strip()
        if extract_out:
            log_lines.append(extract_out)
            _log_queue.put(extract_out)
        if extract_result.returncode != 0 or not os.path.exists(lstm_source):
            err = f"[ERROR] Failed to extract {lstm_source} from {traineddata_for_extract}"
            log_lines.append(err)
            _log_queue.put(err)
            full_log = "\n".join(log_lines)
            try:
                with db_engine.connect() as sync_conn:
                    with sync_conn.begin():
                        sync_conn.execute(
                            text("UPDATE training_runs SET status='failed', completed_at=:t, log=:log WHERE id=:id"),
                            {"t": datetime.utcnow(), "log": full_log, "id": run_id},
                        )
            except Exception:
                pass
            _is_training = False
            _active_run_id = None
            _log_queue.put("[DONE]")
            return

    try:
        # Use tessdata_best traineddata for training
        traineddata_path = os.path.join(tessdata_best_prefix, f"{base_sin_model}.traineddata")
        cmd = [
            lstmtraining_path,
            "--continue_from", lstm_source,
            "--model_output", output_base,
            "--traineddata", traineddata_path,
            "--train_listfile", lstmf_list_file,
            "--max_iterations", str(iterations),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            line = line.rstrip()
            log_lines.append(line)
            _log_queue.put(line)

        proc.wait()

        if proc.returncode == 0:
            active_dir = os.path.join(storage_path, "models", "active")
            os.makedirs(active_dir, exist_ok=True)

            # output base for traineddata (no extension — lstmtraining appends it)
            final_output_base = os.path.join(active_dir, "sin_id")

            # Use tessdata_best traineddata for stop_training step
            traineddata_source = os.path.join(tessdata_best_prefix, f"{base_sin_model}.traineddata")

            # Find the best checkpoint dynamically
            # lstmtraining writes: sin_id_<error>_<iter>_<total>.checkpoint
            # The rolling sin_id.checkpoint is sometimes missing, so find the
            # best one by picking the file with the lowest BCER error rate.
            checkpoint_files = glob.glob(f"{output_base}_*.checkpoint")

            if not checkpoint_files:
                err = f"[ERROR] No checkpoint files found in {run_dir}. Training produced no output."
                log_lines.append(err)
                _log_queue.put(err)
            else:
                # Filename format: sin_id_<error>_<iter>_<total>.checkpoint
                # e.g. sin_id_7.764_208_400.checkpoint
                # parts after split('_'): ['sin', 'id', '7.764', '208', '400']
                def extract_error_rate(path: str) -> float:
                    try:
                        basename = os.path.basename(path)
                        parts = basename.replace(".checkpoint", "").split("_")
                        return float(parts[2])
                    except (IndexError, ValueError):
                        return float("inf")

                checkpoint = min(checkpoint_files, key=extract_error_rate)
                best_error = extract_error_rate(checkpoint)

                msg = f"Using best checkpoint: {os.path.basename(checkpoint)} (BCER={best_error}%)"
                log_lines.append(msg)
                _log_queue.put(msg)

                stop_cmd = [
                    lstmtraining_path,
                    "--stop_training",
                    "--continue_from", checkpoint,
                    "--traineddata", traineddata_source,
                    "--model_output", final_output_base,
                ]

                stop_result = subprocess.run(stop_cmd, capture_output=True, text=True)
                stop_out = (stop_result.stdout + stop_result.stderr).strip()
                if stop_out:
                    log_lines.append(stop_out)
                    _log_queue.put(stop_out)

                if stop_result.returncode == 0:
                    # lstmtraining --stop_training should write <model_output>.traineddata
                    produced = f"{final_output_base}.traineddata"

                    # Log all files in active_dir so we can always see what was produced
                    active_files = os.listdir(active_dir)
                    _log_queue.put(f"Files in active dir after stop_training: {active_files}")
                    log_lines.append(f"Files in active dir after stop_training: {active_files}")

                    if not os.path.exists(produced):
                        # Some Tesseract versions write the file without the .traineddata
                        # extension (e.g. just 'sin_id'). Detect and rename it.
                        bare_file = final_output_base  # e.g. .../active/sin_id
                        if os.path.exists(bare_file) and not os.path.isdir(bare_file):
                            os.rename(bare_file, produced)
                            msg = (
                                f"Renamed bare output "
                                f"'{os.path.basename(bare_file)}' → "
                                f"'{os.path.basename(produced)}'"
                            )
                            log_lines.append(msg)
                            _log_queue.put(msg)
                        else:
                            # Last resort: pick any .traineddata in active_dir
                            candidates = glob.glob(os.path.join(active_dir, "*.traineddata"))
                            if candidates:
                                produced = max(candidates, key=os.path.getmtime)
                                msg = f"Expected output not found, using: {os.path.basename(produced)}"
                                log_lines.append(msg)
                                _log_queue.put(msg)
                            else:
                                err = (
                                    f"[ERROR] stop_training succeeded (rc=0) but no output found "
                                    f"in {active_dir}. Files present: {active_files}"
                                )
                                log_lines.append(err)
                                _log_queue.put(err)
                                produced = None

                    if produced and os.path.exists(produced):
                        # Copy finished model to tessdata_prefix so Tesseract picks it up immediately
                        dest = os.path.join(tessdata_prefix, "sin_id.traineddata")
                        shutil.copy2(produced, dest)
                        final_model_path = produced
                        success = True
                        _log_queue.put(f"Model successfully saved to: {dest}")
                        log_lines.append(f"Model successfully saved to: {dest}")
                else:
                    err = f"[ERROR] stop_training failed with return code {stop_result.returncode}"
                    log_lines.append(err)
                    _log_queue.put(err)

    except Exception as exc:
        err = f"[ERROR] {exc}"
        log_lines.append(err)
        _log_queue.put(err)

    full_log = "\n".join(log_lines)
    status = "completed" if success else "failed"

    try:
        with db_engine.connect() as sync_conn:
            with sync_conn.begin():
                sync_conn.execute(
                    text(
                        "UPDATE training_runs "
                        "SET status=:status, completed_at=:completed_at, "
                        "file_count=:file_count, model_path=:model_path, log=:log "
                        "WHERE id=:id"
                    ),
                    {
                        "status": status,
                        "completed_at": datetime.utcnow(),
                        "file_count": file_count,
                        "model_path": final_model_path,
                        "log": full_log,
                        "id": run_id,
                    },
                )
                if success:
                    sync_conn.execute(
                        text("UPDATE training_runs SET is_active=FALSE WHERE id != :id"),
                        {"id": run_id},
                    )
                    sync_conn.execute(
                        text("UPDATE training_runs SET is_active=TRUE WHERE id=:id"),
                        {"id": run_id},
                    )
    except Exception as exc:
        _log_queue.put(f"[DB ERROR] {exc}")

    _is_training = False
    _active_run_id = None
    _log_queue.put("[DONE]")