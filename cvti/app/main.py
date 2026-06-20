"""Desktop application entry point."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("PyQt6 is required: pip install PyQt6")
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
