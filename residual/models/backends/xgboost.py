from __future__ import annotations

import sys
import plugins.residual.backends.xgboost as _impl

sys.modules[__name__] = _impl
