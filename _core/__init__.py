"""Internal reproducibility kernels (not a general-purpose solver API)."""
import os
import tempfile
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'di_bregman_mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'di_bregman_cache'))
os.environ.setdefault('MPLBACKEND', 'Agg')
