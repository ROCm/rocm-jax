import os
import os.path
import re
import sys

g_gdb_available = False
try:
    import gdb

    g_gdb_available = True
except:  # noqa
    print("GDB extension not available, all symbols might not be available")

# pip install perfetto
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import (
    TrackEvent,
    TrackDescriptor,
    ProcessDescriptor,
    ThreadDescriptor,
)

g_sym_file = "sym.log"
g_trace_file = "trace.pftrace"

g_work_dir = os.path.dirname(__file__)
g_sym_filepath = g_work_dir + "/" + g_sym_file

g_outdir = g_work_dir
g_trace_filepath = g_outdir + "/" + g_trace_file


def get_symbols(addresses: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """Obtain symbols for given set of addresses"""
    if not g_gdb_available:
        return addresses

    def _isHex(val: str):
        if val.startswith("0x"):
            try:
                v = int(val, 16)  # noqa
                return True
            except:  # noqa
                pass
        return False

    r_sym = re.compile(r"^(.+)\s+in section (.*)$")
    ret: dict[str, tuple[str, str]] = {}
    for addr in addresses.keys():
        resp = gdb.execute(f"info symbol 0x{addr}", to_string=True)
        if not isinstance(resp, str):
            raise RuntimeError(f"For '{addr}' GDB returned '{resp}'")
        if m := r_sym.match(resp):
            assert addr not in ret
            sym = m.group(1)
            oldsym = addresses[addr][0]
            assert _isHex(oldsym) or oldsym == sym, (
                "GDB symbol doesn't match the probe!"
            )
            ret[addr] = (sym, m.group(2))
        else:
            raise RuntimeError(f"For '{addr}' can't match GDB responce '{resp}'")
    return ret


def parse_log(logfile) -> tuple[dict[int, list[list]], dict[str, tuple[str, str]]]:
    """parses log into a dict of per-thread calls and symbols description"""
    if not os.path.exists(logfile):
        raise RuntimeError(f"logfile {logfile} not found")

    # spaces are added to let slighlty modified logs remain parsable
    r_entry_all = re.compile(
        r"^\s*uprobe:[^:]+:\s*(?P<probe>\w+),\s*(?P<addr>[0-9a-fA-F]+)\s*,\s*(?P<tid>\d+)\s*,\s*(?P<ts_start>[0-9a-fA-F]+)\s*"
    )
    r_entry_AllRedScat = re.compile(
        r",\s*(?P<size>[0-9a-fA-F]+)\s*,\s*(?P<dtype>[0-9a-fA-F]+)\s*,\s*(?P<red_op>[0-9a-fA-F]+)\s*,\s*(?P<comm>[0-9a-fA-F]+)$"
    )
    r_entry_AllGath = re.compile(
        r",\s*(?P<sendcount>[0-9a-fA-F]+)\s*,\s*(?P<dtype>[0-9a-fA-F]+)\s*,\s*(?P<comm>[0-9a-fA-F]+)$"
    )
    r_entry_InitRankCfg = re.compile(
        r",\s*(?P<nranks>[0-9a-fA-F]+)\s*,\s*(?P<rank>[0-9a-fA-F]+)$"
    )
    r_entry_split = re.compile(
        r",\s*(?P<comm>[0-9a-fA-F]+)\s*,\s*(?P<color>[0-9a-fA-F]+)\s*,\s*(?P<key>[0-9a-fA-F]+)$"
    )

    entry_special = frozenset(
        [
            "ncclAllReduce",
            "ncclReduceScatter",
            "ncclAllGather",
            "ncclCommInitRankConfig",
            "ncclCommSplit",
        ]
    )

    def _getEntryRegExp(probe: str):
        if probe in entry_special:
            if probe in ["ncclAllReduce", "ncclReduceScatter"]:
                return r_entry_AllRedScat
            elif probe == "ncclAllGather":
                return r_entry_AllGath
            elif probe == "ncclCommInitRankConfig":
                return r_entry_InitRankCfg
            elif probe == "ncclCommSplit":
                return r_entry_split
            else:
                assert False
        return None

    r_exit_all = re.compile(
        r"^\s*uretprobe:[^:]+:\s*(?P<probe>\w+),\s*(?P<caller_addr>[0-9a-fA-F]+)\s*,\s*(?P<callee_addr>[0-9a-fA-F]+)\s*,\s*(?P<tid>\d+)\s*,\s*(?P<ts_end>[0-9a-fA-F]+)\s*"
    )
    r_exit_InitRankCfgSplit = re.compile(r",\s*(?P<newcomm>[0-9a-fA-F]+)$")

    # mapping from an address to a symbol to use if GDB isn't available
    addresses: dict[str, tuple[str, str]] = {}

    def _updateAddresses(addr: str, val: str):
        assert isinstance(val, str)
        if addr in addresses:
            assert val == addresses[addr][0]
        else:
            addresses[addr] = (val, "<n/a>")

    # maps tid -> [ [start_idx, addr, ts_start, ts_end, caller_addr, suppl] ]
    calls: dict[int, list[list]] = {}

    with open(logfile, "r") as file:
        for idx, line in enumerate(file):
            idx += 1  # zero idx adjust
            if m_entry := r_entry_all.match(line):
                probe = m_entry["probe"]
                addr = m_entry["addr"]  # should be left as a string
                assert addr != "0"
                tid = int(m_entry["tid"])
                assert tid > 0
                ts_start = int(m_entry["ts_start"], 16)
                assert ts_start > 0

                _updateAddresses(addr, probe)

                if tid in calls:
                    _, p_addr, p_ts_start, p_ts_end, p_caller, _ = calls[tid][-1]
                    assert p_ts_start < ts_start

                assert tid not in calls or (0 < p_ts_end and p_caller is not None)

                call = [idx, addr, ts_start, 0, None, {"_probe": probe}]
                calls.setdefault(tid, []).append(call)

                rest_line = line[len(m_entry.group(0)) :].rstrip()
                reg = _getEntryRegExp(probe)
                if reg is None:
                    assert not rest_line, (
                        f"Parser isn't coherent with the log, rest_line='{rest_line}', so can't parse #{idx} {line}"
                    )
                else:
                    m = reg.match(rest_line)
                    assert m
                    call[-1].update({k: int(v, 16) for k, v in m.groupdict().items()})

            elif m_exit := r_exit_all.match(line):
                probe = m_exit["probe"]
                caller_addr = m_exit["caller_addr"]  # should be left as a string
                assert caller_addr != "0"
                callee_addr = m_exit["callee_addr"]  # should be left as a string
                assert callee_addr != "0", "The log is broken, callee address isn't set"
                tid = int(m_exit["tid"])
                assert tid > 0
                ts_end = int(m_exit["ts_end"], 16)
                assert ts_end > 0

                _updateAddresses(caller_addr, f"0x{caller_addr}")
                _updateAddresses(callee_addr, probe)

                _, addr, ts_start, old_ts_end, old_caller, suppl = calls[tid][-1]
                assert probe == suppl["_probe"]
                assert callee_addr == addr
                assert old_ts_end == 0 and old_caller is None
                assert ts_start < ts_end, (
                    f"Line#{idx}, ts_end={ts_end} >= ts_start={ts_start}"
                )

                calls[tid][-1][3] = ts_end  # marking as not running
                calls[tid][-1][4] = caller_addr

                rest_line = line[len(m_exit.group(0)) :].rstrip()
                if probe in ["ncclCommInitRankConfig", "ncclCommSplit"]:
                    m = r_exit_InitRankCfgSplit.match(rest_line)
                    assert m
                    suppl.update({k: int(v, 16) for k, v in m.groupdict().items()})
                else:
                    assert not rest_line, (
                        f"Parser isn't coherent with the log, rest_line='{rest_line}', so can't parse #{idx} {line}"
                    )

            elif line:
                raise RuntimeError(f"Unparsable line#{idx} {line}")

    return calls, get_symbols(addresses)


"""   NOT UPDATED v1->v2, calls format has changed
def report_single_overlaps(
    symbols: dict[str, tuple[str, str]], calls: dict[int, list[list]]
):
    print("Report on unique individual call overlaps begin ----->")
    symbol = lambda x: symbols[x][0]  # noqa: E731
    make_call_spec = lambda i, t, a, d: f"#{i} {t}:{symbol(a)} ({d:.0f}us)"  # noqa: E731

    pair_reported: set[tuple[int, int]] = set()

    for tid, call_list in calls.items():
        thread_report = False
        prev_ts_end = 0
        for vals in call_list:
            idx, addr, ts_start, ts_end, caller_addr = vals
            assert 0 < ts_end
            assert prev_ts_end < ts_start
            assert ts_start < ts_end
            prev_ts_end = ts_end
            dur_us = (ts_end - ts_start) / 1000.0
            call_spec = None

            for otid, ocall_list in calls.items():
                if tid == otid:
                    continue
                for ovals in ocall_list:
                    oidx, oaddr, ots_start, ots_end, ocaller_addr = ovals

                    pair_key = (min(idx, oidx), max(idx, oidx))
                    if pair_key in pair_reported:
                        continue

                    if ots_end <= ts_start:
                        continue  # too early
                    osize_us = (
                        min(ts_end, ots_end) - max(ts_start, ots_start)
                    ) / 1000.0
                    if osize_us > 0:
                        if not thread_report:
                            thread_report = True
                            print(f"  based on thread {tid}:")
                        if call_spec is None:
                            call_spec = make_call_spec(idx, tid, addr, dur_us)
                        odur_us = (ots_end - ots_start) / 1000.0
                        ocall_spec = make_call_spec(oidx, otid, oaddr, odur_us)
                        print(
                            f"{call_spec} overlaps {ocall_spec} by {osize_us:.0f}us ({100.0 * osize_us / dur_us:.1f}% of this and {100.0 * osize_us / odur_us:.1f}% of other)"
                        )
                        pair_reported.add(pair_key)
                    else:
                        break  # too late, no need to look further

    print(f"-----> end of report, {len(pair_reported)} overlaps found")
"""


class MyTraceBuilder:
    def __init__(self):
        self.builder = TraceProtoBuilder()

    def make_trace(
        self,
        tracepath: str,
        symbols: dict[str, tuple[str, str]],
        calls: dict[int, list[list]],
    ):
        print(f"\nCreating trace {tracepath} ...")
        self._make_trace_packets(symbols, calls)

        with open(tracepath, "wb") as f:
            f.write(self.builder.serialize())

        print(f"Trace written to {tracepath}")
        print("Open with https://magic-trace.org/ or https://ui.perfetto.dev")

    def _make_trace_packets(
        self,
        symbols: dict[str, tuple[str, str]],
        calls: dict[int, list[list]],
    ):
        builder: TraceProtoBuilder = self.builder

        symbol = lambda x: symbols[x][0]  # noqa: E731

        def _makeEventName(addr: str) -> str:
            sym = symbol(addr)
            if sym.startswith("nccl"):
                return sym[4:]
            return sym

        def _add_annotation(packet, dict_or_name, value=None):
            if isinstance(dict_or_name, str):
                assert value is not None
                dict_or_name = {dict_or_name: value}
            assert isinstance(dict_or_name, dict)

            for name, value in dict_or_name.items():
                assert isinstance(name, str)
                if name.startswith("_"):
                    continue

                value = convSuppl(name, value)

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = name

                # Set the appropriate value field based on type
                if isinstance(value, bool):
                    annotation.bool_value = value
                elif isinstance(value, int):
                    annotation.int_value = value
                elif isinstance(value, float):
                    annotation.double_value = value
                elif isinstance(value, str):
                    annotation.string_value = value
                else:
                    raise RuntimeError(
                        "Unsupported annotation type " + str(type(value))
                    )

        # Define a unique ID for this sequence of packets (generate once per trace producer)
        PACKET_SEQUENCE_ID = 0  # Choose any unique integer

        for tid, call_list in calls.items():
            # Define a unique UUID for your custom track (generate a 64-bit random number)
            CUSTOM_TRACK_UUID = tid

            # 1. Define the Custom Track
            # This packet describes the track on which your events will be displayed.
            # Emit this once at the beginning of your trace.
            packet = builder.add_packet()
            packet.track_descriptor.uuid = CUSTOM_TRACK_UUID
            packet.track_descriptor.name = f"Thread {tid}"

            for vals in call_list:
                idx, addr, ts_start, ts_end, caller_addr, suppl = vals

                # 2. Emit events for this custom track
                packet = builder.add_packet()
                packet.timestamp = ts_start  # Start time in nanoseconds
                packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
                # Associate with the track
                packet.track_event.track_uuid = CUSTOM_TRACK_UUID
                # packet.track_event.name = f"#{idx} {symbol(addr)}"
                packet.track_event.name = _makeEventName(addr)
                packet.track_event.correlation_id = int(addr, 16)
                packet.trusted_packet_sequence_id = PACKET_SEQUENCE_ID

                flow_ids = []
                v = suppl.get("newcomm")
                if v:
                    flow_ids.append(v)
                v = suppl.get("comm")
                if v:
                    flow_ids.append(v)
                for f in flow_ids:
                    packet.track_event.flow_ids.append(f)

                _add_annotation(packet, "Log line number", idx)
                _add_annotation(
                    packet,
                    "Called from",
                    f"{symbol(caller_addr)} @ {symbols[caller_addr][1]}",
                )
                _add_annotation(packet, suppl)

                packet = builder.add_packet()
                packet.timestamp = ts_end  # End time in nanoseconds
                packet.track_event.type = TrackEvent.TYPE_SLICE_END
                packet.track_event.track_uuid = CUSTOM_TRACK_UUID
                packet.trusted_packet_sequence_id = PACKET_SEQUENCE_ID


def convSuppl(key: str, value):
    if key in ("comm", "newcomm"):
        return "0x%x" % value
    return value


def print_per_thread(symbols: dict[str, tuple[str, str]], calls: dict[int, list[list]]):
    # finding the earliest timestamp
    min_ts = min(min(vals[2], vals[3]) for clist in calls.values() for vals in clist)

    symbol = lambda x: symbols[x][0]  # noqa: E731

    print("Per thread calls begin ----->")
    print("log_line_num, func, ts_start_ns, ts_end_ns, duration_ns, supplement, caller")
    for tid, call_list in calls.items():
        print(f"Thread {tid}:")
        prev_ts_end = 0
        for vals in call_list:
            idx, addr, ts_start, ts_end, caller_addr, suppl = vals
            assert 0 < ts_end
            assert prev_ts_end < ts_start
            assert ts_start < ts_end
            prev_ts_end = ts_end
            s = {k: convSuppl(k, v) for k, v in suppl.items() if not k.startswith("_")}
            print(
                f" #{idx}, {symbol(addr)}, {ts_start - min_ts}, {ts_end - min_ts}, {ts_end - ts_start}, {s}, {symbol(caller_addr)}"
            )
    print("-----> end of per thread calls")

    print("Communicators report ------>")
    # flap map communicator -> (tid, line_num, ts_start) for tracking
    comms: dict[int, tuple[int, int, int]] = {}
    created_comms = set()
    # first just gather all ensuring no overwrites are happening, then validate
    for tid, call_list in calls.items():
        for vals in call_list:
            idx, addr, ts_start, _, _, suppl = vals
            v = suppl.get("newcomm")
            if v is not None and v:
                created_comms.add(v)
                if v in comms:
                    ots = comms[v][2]
                    if ts_start < ots:  # here is the proper init moment
                        comms[v] = (tid, idx, ts_start)
                else:
                    comms[v] = (tid, idx, ts_start)

            v = suppl.get("comm")
            if v is not None and v:
                if v in comms:
                    otid, oidx, ots_start = comms[v]
                    if ts_start < ots_start:
                        comms[v] = (tid, idx, ts_start)
                else:
                    comms[v] = (tid, idx, ts_start)

    for tid, call_list in calls.items():
        for vals in call_list:
            idx, addr, ts_start, _, _, suppl = vals

            v = suppl.get("newcomm")
            if v is not None:
                if v:
                    ots = comms[v][2]
                    if ts_start > ots:  # here is the proper init moment
                        print(
                            f"!!! WARNING: #{idx} {symbol(addr)} (tid={tid}) created the same newcomm={v:x} "
                            f"as was created by #{comms[v][1]} (tid={comms[v][0]})."
                            "This is a bug in program or in logging!"
                        )
                    elif ts_start == ots:
                        if tid != comms[v][0]:
                            print(
                                f"!!! WARNING: #{idx} {symbol(addr)} (tid={tid}) created the same newcomm={v:x} "
                                f"at the same time as was created by #{comms[v][1]} (tid={comms[v][0]})."
                                "This is a bug in program or in logging!"
                            )
                else:
                    print(
                        f"WARNING: #{idx} {symbol(addr)} (tid={tid}) didn't create a valid newcomm"
                    )

            v = suppl.get("comm")
            if v is not None:
                if v:
                    otid, oidx, ots_start = comms[v]
                    if tid != otid:
                        print(
                            f"#{idx} {symbol(addr)} (tid={tid}) uses comm={v:x} that was "
                            f"created by a different thread {otid} #{oidx}."
                        )
                else:
                    print(
                        f"WARNING: #{idx} {symbol(addr)} (tid={tid}) uses invalid comm!"
                    )
    unknown = created_comms - comms.keys()
    if unknown:
        print("The following communicators are used, but never created:")
        for k in sorted(list(unknown)):
            c = comms[k]
            print(f"   #{c[1]} tid={c[0]}")
    print("------> communicators report")


def process_log(logfile, tracepath):
    print(f"Using log file {logfile}")

    calls, symbols = parse_log(logfile)

    # report_single_overlaps(symbols, calls)
    print_per_thread(symbols, calls)

    print("address -> symbol mapping:")
    for addr, descr in symbols.items():
        print(f"{addr} := {descr[0]} @ {descr[1]}")

    MyTraceBuilder().make_trace(tracepath, symbols, calls)


def getFiles() -> tuple[str, str]:
    logfile = g_sym_filepath
    trace = g_trace_filepath
    if len(sys.argv) >= 2:
        logfile = sys.argv[1]
        if not os.path.isabs(logfile):
            logfile = g_work_dir + "/" + logfile

        root,_ = os.path.splitext(os.path.basename(logfile))
        trace = os.path.dirname(logfile) + f"/{root}.pftrace"

    return logfile, trace


if __name__ == "__main__":
    process_log(*getFiles())
