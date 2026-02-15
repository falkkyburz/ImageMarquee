uv run pyinstaller --onefile --windowed --name "ImageMarquee" \
  --icon=image_marquee.ico \
  --exclude-module QtWebEngine \
  --exclude-module QtDesigner \
  --exclude-module QtQml \
  --exclude-module QtQuick \
  --exclude-module QtNetwork \
  image_marquee.py
