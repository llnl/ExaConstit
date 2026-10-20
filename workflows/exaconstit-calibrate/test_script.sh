  PY=/usr/tce/packages/python/python-3.12.2/bin/python
  export PYTHONPATH=/usr/lib64/flux/python3.12${PYTHONPATH:+:$PYTHONPATH}

  $PY -m pip install -e ".[test,nsga3,plot]"
  cd examples
  srun -n 1 --pty --mpi=none --mpibind=off flux start $PY nsga3_calibration.py --backend flux