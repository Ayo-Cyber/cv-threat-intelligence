"""Desktop application entry point."""

from __future__ import annotations

import sys
from cvti.logging_setup import get_logger

log = get_logger(__name__)


def main() -> None:
    # Entrypoint: configure logging before anything can fail.
    from cvti.logging_setup import setup_logging
    setup_logging(component="argus-app")
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        log.info("PyQt6 is required: pip install PyQt6")
        sys.exit(1)

    from cvti.app.window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("CV Threat Intelligence")
    app.setOrganizationName("CVTI")

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
