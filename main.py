import subprocess
import sys


def main():
    """Launch the Streamlit web interface."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
