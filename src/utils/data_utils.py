from copy import deepcopy
import os


import os
import shutil
import socket
import subprocess
from pathlib import Path


class ExperimentWorkspace:
    def __init__(self, save_base_path, run_name, overwrite=False, use_condor=False):
        """
        Args:
            save_base_path (str/Path): The root directory where experiments are saved.
            run_name (str): The full specific name of this run.
            overwrite (bool): If True, delete broken/existing runs of the same name.
            use_condor (bool): If True, use HTCondor scratch space.
        """
        self.save_base_path = Path(save_base_path).resolve()
        self.run_name = run_name
        self.overwrite = overwrite
        self.use_condor = use_condor
        
        # This is the final permanent destination (e.g. shared storage)
        self.permanent_dir = self.save_base_path / self.run_name
        print("DEBUG: permanent_dir", self.permanent_dir)
        
        # This is where we actually write files (might be temp scratch)
        self.working_dir = self._determine_working_dir()
        
        # Sub-directories
        self.eval_dir = self.working_dir / 'results_summary'


    @property
    def is_completed(self):
        """Checks if a run exists and was finished successfully (no broken.txt)."""
        # We check the permanent dir for completion status
        if not self.permanent_dir.exists():
            return False
        return not (self.permanent_dir / "broken.txt").exists()


    def setup(self):
        """Prepares directories. Returns True if we should proceed, False if run exists."""
        
        # 1. Check if run already exists
        if self.is_completed:
            if not self.overwrite:
                print(f"Run {self.run_name} already completed. Skipping.")
                return False
            else:
                print(f"Run completed, but overwrite is ON. Cleaning {self.permanent_dir}...")
                shutil.rmtree(self.permanent_dir, ignore_errors=True)

        # 2. Handle broken/partial runs
        self._clean_partial_matches()

        # 3. Create directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # 4. Mark run as "in progress"
        self._set_status_broken(True)
        
        print(f"Workspace setup at: {self.working_dir}")
        return True


    def cleanup(self):
        """Call this at the end of the script."""
        # 1. Mark run as complete (remove broken.txt)
        self._set_status_broken(False)
        
        # 2. Sync back if using Condor
        if self.use_condor:
            self._sync_condor_results()


    
    def _determine_working_dir(self):
        if not self.use_condor:
            return self.permanent_dir
        
        scratch_dir = os.environ.get('_CONDOR_SCRATCH_DIR')
        if not scratch_dir:
            print("_CONDOR_SCRATCH_DIR not found. Falling back to local storage.")
            return self.permanent_dir
            
        # Replicate the last 2 folder segments to avoid name collisions in scratch
        # e.g., scratch/experiment_group/specific_run_name
        rel_path = Path(self.save_base_path.name) / self.run_name
        return Path(scratch_dir) / rel_path

    def _clean_partial_matches(self):
        """Logic to find and remove directories that match the run name (handling the _e= suffix logic)."""
        if not self.overwrite:
            return

        # Canonical name (stripping _e=...)
        canonical_name = self.run_name.split('_e=')[0] if '_e=' in self.run_name else self.run_name

        if not self.save_base_path.exists():
            return

        # Scan parent directory
        for p in self.save_base_path.iterdir():
            if not p.is_dir(): 
                continue
            
            # If directory contains our canonical name
            if canonical_name in p.name:
                is_broken = (p / "broken.txt").exists()
                # If we are strictly overwriting, or if the found run is broken
                if self.overwrite or is_broken:
                    print(f"Removing existing/broken run: {p}")
                    shutil.rmtree(p, ignore_errors=True)

    def _set_status_broken(self, is_broken: bool):
        """Manages the broken.txt marker."""
        marker = self.working_dir / "broken.txt"
        if is_broken:
            marker.touch(exist_ok=True)
        else:
            marker.unlink(missing_ok=True)

    def _sync_condor_results(self):
        """Copies results from scratch back to permanent storage via SCP."""
        print("Syncing Condor results back to permanent storage...")
        
        hostname = socket.gethostname().split(".")[0]
        # Ensure parent of permanent dir exists
        self.permanent_dir.parent.mkdir(parents=True, exist_ok=True)

        # Build SCP command
        cmd = [
            "scp", "-4",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-r",
            f"{hostname}:{self.working_dir}",
            str(self.permanent_dir)
        ]
        
        try:
            print("Running command:", cmd)
            subprocess.check_call(cmd)
            print("Sync complete.")
        except subprocess.CalledProcessError as e:
            # print(f"Failed to sync results: {e}")
            print(f"SCP failed: {e}")
            print("Falling back to rsync copy...")
            
            # Direct rsync (works when paths are locally accessible)
            rsync_cmd = f"rsync -avh --progress {self.working_dir}/ {self.permanent_dir}/"
            result = subprocess.run(rsync_cmd, shell=True)
            if result.returncode == 0:
                print("Rsync sync complete.")
            else:
                print(f"Rsync also failed with exit code: {result.returncode}")
                raise RuntimeError("Failed to sync results with both SCP and rsync.")
        
        # ssh_connector = socket.gethostname().split(".")[0]
        # # Ensure parent of permanent dir exists
        # self.permanent_dir.parent.mkdir(parents=True, exist_ok=True)
        # print("\nssh connector", ssh_connector)
        # print("\ncondor_results_dir:", self.working_dir) # condor_results_dir
        # print("\nresults_dir:", self.permanent_dir) # results_dir
        # print("\nscp -4 -r {}:{} {}".format(ssh_connector, self.working_dir, self.permanent_dir))
        # # Copy the contents of the condor_results_dir to the results_dir
        # os.system("scp -4 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r {}:{} {}".format(ssh_connector, self.working_dir, self.permanent_dir))
        # print("complete...")
        
        