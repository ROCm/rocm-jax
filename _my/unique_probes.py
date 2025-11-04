import re
import os
import sys


g_work_dir = os.path.dirname(__file__)


def process_log(logfile):
    r_entry = re.compile(r"^\s*uprobe:[^:]+:\s*(?P<probe>\w+)")
    r_exit = re.compile(r"^\s*uretprobe:[^:]+:\s*(?P<probe>\w+)")

    probes=set()

    with open(logfile, "r") as file:
        for idx, line in enumerate(file):
            if m_entry := r_entry.match(line):
                probe = m_entry["probe"]
            elif m_exit := r_exit.match(line):
                probe = m_exit["probe"]
            else:
                raise RuntimeError(f"Unparsable line#{idx} {line}")
            probes.add(probe)

    print("\n".join([f"{i:2d} {p}" for i,p in enumerate(sorted(list(probes)))]))

def getLogFile() -> str:
    if len(sys.argv) < 2:
        raise RuntimeError("First argument must be a path to log file")
    file = sys.argv[1]
    if not os.path.isabs(file):
        file = g_work_dir + "/" + file
    return file


if __name__ == "__main__":
    process_log(getLogFile())
