from __future__ import annotations

import sys
import plugins.residual.backends.lightgbm as _impl

sys.modules[__name__] = _impl
