import sys


def progress_display(progress, total, extra_info=""):
    percent = (progress / total) * 100
    sys.stdout.write(f"\rProgresso: {progress}/{total} ({percent:.2f}%) {extra_info}")
    sys.stdout.flush()
