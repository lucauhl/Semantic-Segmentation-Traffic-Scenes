# Cluster Cheatsheet (Alex)

## Jobs (SLURM)
- Submit: `sbatch submit_job.sh`
- Status/start: `squeue --me` / `squeue --me --start`
- Cancel: `scancel <JOB_ID>` (all: `scancel -u <USER>`)

## Logs
- Live output: `tail -f logs/output_*.txt`
- Errors: `cat logs/error_*.txt`
- List logs: `ls -l logs/`

## Environment / Install
- Load Python: `module load python/pytorch-1.10py3.9`
- Install package: `pip install --user <pkg>`
- Show user packages: `pip list --user`


