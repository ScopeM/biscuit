import sys
import os

# Set the backend ONLY if pyplot has not yet been imported.
# This avoids runtime errors when another module (e.g. in a notebook) has already
# selected a backend. The backend can be overridden via the environment variable
# MATPLOTLIB_BACKEND. If the variable is not set and no backend has been chosen
# yet, we fall back to "TkAgg" for interactive local sessions.
if 'matplotlib.pyplot' not in sys.modules:
    #backend = os.environ.get('MATPLOTLIB_BACKEND', 'TkAgg')
    import matplotlib
    #matplotlib.use(backend)
else:
    import matplotlib
