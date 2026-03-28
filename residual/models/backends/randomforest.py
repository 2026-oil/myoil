from __future__ import annotations

import sys
import plugins.residual.backends.randomforest as _impl

sys.modules[__name__] = _impl
