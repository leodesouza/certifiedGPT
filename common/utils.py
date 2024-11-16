# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import os
from common.registry import registry


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


