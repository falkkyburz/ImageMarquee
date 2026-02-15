uv run pyinstaller --onefile --windowed --name "ImageMarquee" --icon=image_marquee.ico --hidden-import PySide6.QtOpenGLWidgets --exclude-module QtWebEngine --exclude-module QtDesigner --exclude-module QtQml --exclude-module QtQuick --exclude-module QtNetwork image_marquee.py
pause
