from __future__ import annotations

import sys
from tuning import search_space as _impl

sys.modules[__name__] = _impl
