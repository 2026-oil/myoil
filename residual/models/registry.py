from __future__ import annotations

import sys
from plugins.residual import registry as _impl

sys.modules[__name__] = _impl
