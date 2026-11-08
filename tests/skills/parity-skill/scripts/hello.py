import sys

print("hello from parity-skill" + (" " + " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""))
