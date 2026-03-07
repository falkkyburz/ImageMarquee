uv run pyinstaller --onefile --windowed --name "ImageMarquee" --icon=image_marquee.ico --hidden-import PySide6.QtOpenGLWidgets --hidden-import PySide6.QtMultimedia --exclude-module QtWebEngine --exclude-module QtDesigner --exclude-module QtQml --exclude-module QtQuick image_marquee.py
pause
