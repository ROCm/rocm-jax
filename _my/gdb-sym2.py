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

g_sym_file = "log.log"
g_trace_file = "trace.pftrace"

g_work_dir = os.path.dirname(__file__)
g_sym_filepath = g_work_dir + "/" + g_sym_file

g_outdir = g_work_dir
g_trace_filepath = g_outdir + "/" + g_trace_file


# duration of kernel completion we measure is between scheduling launch with
# hipLaunchKernel variation and between executing a host callback placed onto the
# stream right after the kernel.
g_kernel_launch = "hip:LaunchKernel->HostFunc"

PROBE_ID_2_NAME = {
    "AG": "ncclAllGather",
    "AR": "ncclAllReduce",
    "CS": "ncclCommSplit",
    "IC": "ncclCommInitRankConfig",
    "HS": "hipStreamSynchronize",
    "GE": "ncclGroupEnd",
    "GS": "ncclGroupStart",
    "RS": "ncclReduceScatter",
    "L": g_kernel_launch,
}

def isNoExit(probe: str) -> bool:
    return probe == "hipStreamSynchronize"

RED_OP_NAMES = {  # ncclRedOp_t
    0: "Sum",
    1: "Prod",
    2: "Max",
    3: "Min",
    4: "Avg",
}


DTYPE_NAMES = {  # ncclDataType_t
    0: "Int8",
    1: "Uint8",
    2: "Int32",
    3: "Uint32",
    4: "Int64",
    5: "Uint64",
    6: "Float16",
    7: "Float32",
    8: "Float64",
    9: "Bfloat16",
    10: "Float8e4m3",
    11: "Float8e5m2",
}


COLL_TYPES = {  # ncclFunc_t
    0: "Broadcast",
    1: "Reduce",
    2: "AllGather",
    3: "ReduceScatter",
    4: "AllReduce",
    5: "SendRecv",
    6: "Send",
    7: "Recv",
    8: "AllToAllPivot",
}

FUNCNAME_2_COLL_TYPE = {
    "ncclAllReduce": 4,
    "ncclReduceScatter": 3,
    "ncclAllGather": 2,
}


def collectiveHasReduction(coll_type: int) -> bool:
    assert isinstance(coll_type, int)
    return coll_type in [1, 3, 4]


def isHex(val: str) -> bool:
    # if val.startswith("0x"):
    try:
        v = int(val, 16)  # noqa
        return True
    except:  # noqa
        pass
    return False


def translateOpSuppl(suppl: dict) -> dict:
    if "send" in suppl or "recv" in suppl:
        assert "send" in suppl and "recv" in suppl
        assert isinstance(suppl["send"], int) and isinstance(suppl["recv"], int)
        suppl["buffs"] = [(suppl["send"], suppl["recv"])]
        del suppl["send"]
        del suppl["recv"]

    return suppl


def getSymbols(addresses: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """Obtain symbols for a given set of addresses"""
    if not g_gdb_available:
        return addresses

    r_sym = re.compile(r"^(.+)\s+in section (.*)$")
    ret: dict[str, tuple[str, str]] = {}
    for addr in addresses.keys():
        if isHex(addr):
            resp = gdb.execute(f"info symbol 0x{addr}", to_string=True)
            if not isinstance(resp, str):
                raise RuntimeError(f"For '{addr}' GDB returned '{resp}'")
            if m := r_sym.match(resp):
                assert addr not in ret
                sym = m.group(1)
                oldsym = addresses[addr][0]
                assert isHex(oldsym) or oldsym == sym, (
                    "GDB symbol doesn't match the probe!"
                )
                ret[addr] = (sym, m.group(2))
            else:
                raise RuntimeError(f"For '{addr}' can't match GDB responce '{resp}'")
        else:
            ret[addr] = addresses[addr]
    return ret


def parseLog(logfile) -> tuple[dict[int, list[list]], dict[str, tuple[str, str]]]:
    """parses log into a dict of per-thread calls and symbols description"""
    if not os.path.exists(logfile):
        raise RuntimeError(f"logfile {logfile} not found")

    # spaces are added to let slighlty modified logs remain parsable
    r_entry_common = re.compile(
        r"^\s*(?P<probe_id>[A-Z]+)\s+(?P<tid>[0-9a-fA-F]+)\s*,\s*(?P<ts_start>[0-9a-fA-F]+)\s*"
    )

    r_entry2_AllRedScat = re.compile(
        r",\s*(?P<send>[0-9a-fA-F]+)\s*,\s*(?P<recv>[0-9a-fA-F]+)\s*,\s*(?P<count>[0-9a-fA-F]+)\s*"
        r",\s*(?P<dtype>[0-9a-fA-F]+)\s*,\s*(?P<red_op>[0-9a-fA-F]+)\s*,\s*(?P<comm>[0-9a-fA-F]+)\s*,\s*(?P<stream>[0-9a-fA-F]+)\s*,\s*(?P<comm_cudaDev>[0-9a-fA-F]+)\s*$"
    )

    r_entry2_AllGath = re.compile(
        r",\s*(?P<send>[0-9a-fA-F]+)\s*,\s*(?P<recv>[0-9a-fA-F]+)\s*,\s*(?P<count>[0-9a-fA-F]+)\s*"
        r",\s*(?P<dtype>[0-9a-fA-F]+)\s*,\s*(?P<comm>[0-9a-fA-F]+)\s*,\s*(?P<stream>[0-9a-fA-F]+)\s*,\s*(?P<comm_cudaDev>[0-9a-fA-F]+)\s*$"
    )
    r_entry2_InitRankCfg = re.compile(
        r",\s*(?P<nranks>[0-9a-fA-F]+)\s*,\s*(?P<rank>[0-9a-fA-F]+)$"
    )
    r_entry2_split = re.compile(
        r",\s*(?P<comm>[0-9a-fA-F]+)\s*,\s*(?P<color>[0-9a-fA-F]+)\s*,\s*(?P<key>[0-9a-fA-F]+)\s*,\s*(?P<comm_cudaDev>[0-9a-fA-F]+)\s*$"
    )
    # func_tag is used to match to a corresponding completion callback
    r_entry2_LaunchKernel = re.compile(
        r",\s*(?P<func_tag>[0-9a-fA-F]+)\s*,\s*(?P<comm>[0-9a-fA-F]+)\s*,\s*(?P<stream>[0-9a-fA-F]+)\s*,\s*(?P<comm_cudaDev>[0-9a-fA-F]+)\s*,\s*(?P<countRedopDtypeCollt>[0-9a-fA-F]+)\s*,\s*(?P<send>[0-9a-fA-F]+)\s*,\s*(?P<recv>[0-9a-fA-F]+)\s*"
    )
    r_entry2_hipStreamSynchronize = re.compile(
        r",\s*(?P<stream>[0-9a-fA-F]+)$"
    )

    r_LaunchKernel_other_bufs = re.compile(
        r",\s*(?P<send>[0-9a-fA-F]+)\s*,\s*(?P<recv>[0-9a-fA-F]+)\s*"
    )

    entry_special = frozenset(
        [
            "ncclAllReduce",
            "ncclReduceScatter",
            "ncclAllGather",
            "ncclCommInitRankConfig",
            "ncclCommSplit",
            "hipStreamSynchronize",
            # "arechLaunchKernel"
        ]
    )

    def _getEntryRegExp(probe: str):
        if probe in entry_special:
            if probe in ["ncclAllReduce", "ncclReduceScatter"]:
                return r_entry2_AllRedScat
            elif probe == "ncclAllGather":
                return r_entry2_AllGath
            elif probe == "ncclCommInitRankConfig":
                return r_entry2_InitRankCfg
            elif probe == "ncclCommSplit":
                return r_entry2_split
            elif probe == "hipStreamSynchronize":
                return r_entry2_hipStreamSynchronize
            # elif probe == "arechLaunchKernel":
            # return r_entry2_LaunchKernel
            else:
                assert False
        return None

    # will use the same regex for the callback
    r_exit_common = re.compile(
        r"^\s*(?P<probe_id>[a-z]+)\s+(?P<tid>[0-9a-fA-F]+)\s*,\s*(?P<ts_end>[0-9a-fA-F]+)\s*"
    )
    r_exit_Split = re.compile(r",\s*(?P<newcomm>[0-9a-fA-F]+)\s*$")
    r_exit_InitRankCfg = re.compile(r",\s*(?P<newcomm>[0-9a-fA-F]+),\s*(?P<comm_cudaDev>[0-9a-fA-F]+)\s*$")

    def _getExitRegExp(probe: str):
        if probe == "ncclCommSplit":
            return r_exit_Split
        elif probe == "ncclCommInitRankConfig":
            return r_exit_InitRankCfg
        return None

    # mapping from an address to a symbol to use if GDB isn't available. Previously I also dumped
    # return addresses, so this was mainly useful back then. Won't delete it, as it might become useful later
    addresses: dict[str, tuple[str, str]] = {"n/a": ("n/a", "n/a")}

    def _updateAddresses(addr: str, val: str):
        assert isinstance(val, str)
        if addr in addresses:
            assert val == addresses[addr][0]
        else:
            addresses[addr] = (val, "n/a")

    # maps tid -> [ [start_idx, addr, ts_start, ts_end, caller_addr, suppl] ] for regular calls with synchronous exits
    calls: dict[int, list[list]] = {}
    # maps func_tag -> [start_idx, tid, ts_start, ts_end, "n/a", suppl]. At the end of log processing it's merged into calls
    kernels: dict[int, list] = {}
    no_exit_calls: dict[int, list[list]] = {}

    with open(logfile, "r") as file:
        for idx, line in enumerate(file):
            if not line.strip():
                continue
            idx += 1  # zero idx adjust
            m_entry_cmn = r_entry_common.match(line)

            if m_entry_cmn:
                probe_id = m_entry_cmn["probe_id"]
                probe = PROBE_ID_2_NAME[probe_id]
                tid = int(m_entry_cmn["tid"], 16)
                assert tid > 0
                ts_start = int(m_entry_cmn["ts_start"], 16)
                assert ts_start > 0
                total_match_len = len(m_entry_cmn.group(0))

                _updateAddresses(probe, probe)
                rest_line = line[total_match_len:].rstrip()

                if probe == g_kernel_launch:
                    m_launch_kernel = r_entry2_LaunchKernel.match(rest_line)
                    assert m_launch_kernel
                    func_tag = int(m_launch_kernel["func_tag"], 16)
                    assert func_tag != 0
                    comm = int(m_launch_kernel["comm"], 16)
                    assert comm != 0
                    stream = int(m_launch_kernel["stream"], 16)
                    assert stream != 0
                    comm_cudaDev = int(m_launch_kernel["comm_cudaDev"], 16)
                    assert comm_cudaDev >= 0
                    countRedopDtypeCollt = int(
                        m_launch_kernel["countRedopDtypeCollt"], 16
                    )
                    assert countRedopDtypeCollt != 0
                    send = int(m_launch_kernel["send"], 16)
                    assert send != 0
                    recv = int(m_launch_kernel["recv"], 16)
                    assert recv != 0

                    bufs = [(send, recv)]
                    ntasks = func_tag & 0x3
                    rest_line = rest_line[len(m_launch_kernel.group(0)) :].rstrip()

                    for _ in range(ntasks):
                        m = r_LaunchKernel_other_bufs.match(rest_line)
                        assert m
                        bufs.append(
                            (int(m.group("send"), 16), int(m.group("recv"), 16))
                        )
                        rest_line = rest_line[len(m.group(0)) :].rstrip()
                    assert not rest_line

                    count = countRedopDtypeCollt >> 32
                    red_op = (countRedopDtypeCollt >> 16) & 0xFF
                    dtype = (countRedopDtypeCollt >> 8) & 0xFF
                    coll_type = countRedopDtypeCollt & 0xFF

                    suppl = {
                        "_probe": probe,
                        "_func_tag": func_tag,
                        "buffs": bufs,
                        "count": count,
                        "dtype": dtype,
                        "coll_type": coll_type,
                        "comm": comm,
                        "stream": stream,
                        "comm_cudaDev": comm_cudaDev,
                    }
                    if collectiveHasReduction(coll_type):
                        suppl["red_op"] = red_op

                    assert func_tag not in kernels
                    kernels[func_tag] = [idx, tid, ts_start, 0, "n/a", suppl]

                else:  # i.e. probe != g_kernel_launch
                    suppl = {"_probe": probe}
                    if probe in FUNCNAME_2_COLL_TYPE:
                        suppl["coll_type"] = FUNCNAME_2_COLL_TYPE[probe]

                    if tid in calls:
                        _, _, p_ts_start, p_ts_end, p_caller, _ = calls[tid][-1]
                        assert p_ts_start < ts_start

                    assert tid not in calls or (0 < p_ts_end and p_caller is not None)

                    call = [idx, probe, ts_start, 0, None, suppl]

                    if isNoExit(probe):
                        call[4] = "n/a"
                        no_exit_calls.setdefault(tid, []).append(call)
                    else:
                        calls.setdefault(tid, []).append(call)

                    reg = _getEntryRegExp(probe)
                    if reg is None:
                        assert not rest_line, (
                            f"Parser isn't coherent with the log, rest_line='{rest_line}', so can't parse #{idx} {line}"
                        )
                    else:
                        m = reg.match(rest_line)
                        assert m
                        call[-1].update(
                            translateOpSuppl(
                                {k: int(v, 16) for k, v in m.groupdict().items()}
                            )
                        )

            elif m_exit_cmn := r_exit_common.match(line):
                probe_id = m_exit_cmn["probe_id"]
                probe = PROBE_ID_2_NAME[probe_id.upper()]
                tid = int(m_exit_cmn["tid"], 16)
                assert tid > 0
                ts_end = int(m_exit_cmn["ts_end"], 16)
                assert ts_end > 0

                if probe == g_kernel_launch:
                    func_tag = ts_end  # reusing existing parser
                    ts_end = tid
                    ker = kernels[func_tag]
                    assert ker[2] < ts_end
                    ker[3] = ts_end
                else:  # i.e. probe != g_kernel_launch
                    caller_addr = "n/a"
                    # _updateAddresses(caller_addr, caller_addr)
                    _updateAddresses(probe, probe)

                    _, _, ts_start, old_ts_end, old_caller, suppl = calls[tid][-1]
                    assert probe == suppl["_probe"]
                    # assert callee_addr == addr or addr == probe
                    assert old_ts_end == 0 and old_caller is None
                    assert ts_start < ts_end, (
                        f"Line#{idx}, ts_end={ts_end} >= ts_start={ts_start}"
                    )

                    calls[tid][-1][3] = ts_end  # marking as not running
                    calls[tid][-1][4] = caller_addr

                    rest_line = line[len(m_exit_cmn.group(0)) :].rstrip()
                    reg = _getExitRegExp(probe)
                    if reg is None:
                        assert not rest_line, (
                            f"Parser isn't coherent with the log, rest_line='{rest_line}', so can't parse #{idx} {line}"
                        )
                    else:
                        m = reg.match(rest_line)
                        assert m
                        suppl.update({k: int(v, 16) for k, v in m.groupdict().items()})
                    
            elif line:
                raise RuntimeError(f"Unparsable line#{idx} {line}")

    # putting kernel launches into the calls
    for ker in kernels.values():
        tid = ker[1]
        ker[1] = ker[5]["_probe"]
        ts_start = ker[2]  # ts_end COULD BE ZERO!
        if tid in calls:
            # finding where to put the call by ts_start ordering
            clist = calls[tid]
            for i, cvals in enumerate(clist):
                if ts_start < cvals[2]:
                    clist.insert(i, ker)
                    break
            else:
                clist.append(ker)
        else:
            print(
                f"INFO: Thread {tid} for #{ker[0]} {ker[5]['_probe']} is not found in calls, adding a new thread"
            )
            calls[tid] = [ker]

    # merging no exit calls in a correct ordering by ts_start
    for tid, no_exit_call_list in no_exit_calls.items():
        if tid in calls:
            for call in no_exit_call_list:
                clist = calls[tid]
                ts_start = call[2]
                # leaving ending time zero to draw as a marker on the trace
                for i, cvals in enumerate(clist):
                    if ts_start < cvals[2]:
                        clist.insert(i, call)
                        break
                else:
                    clist.append(call)
        else:
            calls[tid] = no_exit_call_list

    return calls, getSymbols(addresses)


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
        **kwargs,
    ) -> None:
        print(f"\nCreating trace {tracepath} ...")
        self._make_trace_packets(symbols, calls, **kwargs)

        with open(tracepath, "wb") as f:
            f.write(self.builder.serialize())

        print(f"Trace written to {tracepath}")
        print("Open with https://magic-trace.org/ only! https://ui.perfetto.dev is buggy and might render some slices length incorrectly!")

    @staticmethod
    def _makeCorrelationId(addr) -> int:
        v = hash(addr)
        if v < 0:
            v += 0x8000000000000000
        return v

    @staticmethod
    def _add_annotation(packet, dict_or_name, value=None):
        if isinstance(dict_or_name, str):
            assert value is not None
            dict_or_name = {dict_or_name: value}
        assert isinstance(dict_or_name, dict)

        for name, value in dict_or_name.items():
            assert isinstance(name, str)
            if name.startswith("_"):
                continue

            # this will output coll_type for individual ops, which is duplication, but that's ok
            value = supplElement2text(name, value)

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
                raise RuntimeError("Unsupported annotation type " + str(type(value)))

    # bit values
    LINK_BY_COMM: int = 1 << 0
    LINK_BY_STREAM: int = 1 << 1
    LINK_BY_DEVICE: int = 1 << 2

    def _make_trace_packets(
        self,
        symbols: dict[str, tuple[str, str]],
        calls: dict[int, list[list]],
        *,
        use_tid_pid: bool = False, # b/c people not aware of ui.perfetto.dev bugs
        link_by: int = LINK_BY_COMM,
        assume_magic_trace: bool = True,  # when True, disable correlation_id that magic-trace doesn't support
    ):
        builder: TraceProtoBuilder = self.builder

        # we need to mark unfinished kernels with the latest timestamp; otherwise
        # we'd have to use a time point representation instead of a time slice
        # finding the latest timestamp
        unfinished_ker_ts = 1 + max(
            max(vals[2], vals[3]) for clist in calls.values() for vals in clist
        )

        symbol = lambda x: symbols[x][0]  # noqa: E731

        _cmn_coll_prms = ["dtype", "count"]

        def _makeEventName(addr: str, suppl: dict) -> str:
            sym = symbol(addr)
            if "coll_type" in suppl:
                has_reduce = collectiveHasReduction(suppl["coll_type"])
                if addr == g_kernel_launch:
                    keys = (
                        ("dtype", "count", "coll_type", "red_op")
                        if has_reduce
                        else ("dtype", "count", "coll_type")
                    )
                else:
                    keys = (
                        ("dtype", "count", "red_op")
                        if has_reduce
                        else ("dtype", "count")
                    )
                args = (
                    "("
                    + ", ".join(str(supplElement2text(k, suppl[k])) for k in keys)
                    + ")"
                )
            else:
                args = ""
            return sym + args

        # Define a unique ID for this sequence of packets (generate once per trace producer)
        PACKET_SEQUENCE_ID = 1  # Choose any unique integer

        # 1. Define the Process Track (Optional, but good for naming the process)
        if use_tid_pid:
            print(
                "\n\nWARNING: Use only https://magic-trace.org to view traces! https://ui.perfetto.dev/ is buggy and renders some slices length incorrectly!\n\n"
            )
            # there's probably a bug somewhere, but I don't know perfetto that well
            packet = builder.add_packet()
            packet.timestamp = 1
            packet.track_descriptor.uuid = 999
            packet.track_descriptor.process.pid = 1
            packet.track_descriptor.process.process_name = "Python"

        unique_track_ids = set[int]()  # for sanity check

        for tid, call_list in calls.items():
            # Define a unique UUID for your custom track (generate a 64-bit random number)
            MAIN_TRACK_UUID = tid
            assert MAIN_TRACK_UUID not in unique_track_ids
            unique_track_ids.add(MAIN_TRACK_UUID)

            # 1. Define the Custom Track
            # This packet describes the track on which your events will be displayed.
            # Emit this once at the beginning of your trace.
            packet = builder.add_packet()
            packet.track_descriptor.uuid = MAIN_TRACK_UUID
            main_track_name = f"Thread {tid}"
            if use_tid_pid:
                packet.timestamp = 2
                packet.track_descriptor.thread.pid = 1
                packet.track_descriptor.thread.tid = tid
                packet.track_descriptor.thread.thread_name = main_track_name
            else:
                packet.track_descriptor.name = main_track_name

            for call_idx, vals in enumerate(call_list):
                idx, addr, ts_start, ts_end, caller_addr, suppl = vals

                flow_ids = []
                if addr == g_kernel_launch:
                    this_event_track_id = suppl["_func_tag"]  # guaranteed to be
                    # unique across tags and should be unique across threads, but lets check this
                    assert this_event_track_id != 0
                    assert this_event_track_id not in unique_track_ids
                    unique_track_ids.add(this_event_track_id)
                    # creating a track for potentially overlapping (with anything) launch
                    packet = builder.add_packet()
                    packet.track_descriptor.uuid = this_event_track_id
                    if use_tid_pid:
                        packet.timestamp = ts_start - 1
                        packet.track_descriptor.thread.pid = 1
                        packet.track_descriptor.thread.tid = tid
                    # leaving the same track name
                    packet.track_descriptor.name = main_track_name

                    if ts_end <= 0:
                        ts_end = unfinished_ker_ts

                    # finding operations that spawned the call to add them to flow_ids
                    # and also to use the same correlation id
                    if not assume_magic_trace:
                        this_correlation_id = None
                        coll_has_reduce = collectiveHasReduction(suppl["coll_type"])
                        key_comp = (
                            (
                                suppl["dtype"],
                                suppl["coll_type"],
                                suppl["count"],
                                suppl["red_op"],
                            )
                            if coll_has_reduce
                            else (suppl["dtype"], suppl["coll_type"], suppl["count"])
                        )
                        for bufs in suppl["buffs"]:
                            match_found = False  # each buffer must be matched
                            for o_vals in call_list[call_idx - 1 :: -1]:  # seeing prev ops
                                o_idx, o_addr, _, _, _, o_suppl = o_vals
                                o_bufs = o_suppl.get("buffs")
                                if o_bufs:
                                    assert (
                                        len(o_bufs) == 1
                                    )  # individual ops use 1 buff only
                                    o_bufs = o_bufs[0]
                                    assert isinstance(o_bufs, tuple)
                                    if bufs == o_bufs:
                                        # testing if other call parameters match
                                        if coll_has_reduce:
                                            match_found = (
                                                o_suppl["dtype"],
                                                o_suppl["coll_type"],
                                                o_suppl["count"],
                                                o_suppl["red_op"],
                                            ) == key_comp
                                        else:
                                            match_found = (
                                                o_suppl["dtype"],
                                                o_suppl["coll_type"],
                                                o_suppl["count"],
                                            ) == key_comp
                                        if match_found:
                                            # idx shouldn't collide with comm ids
                                            # flow_ids.append(idx) # essentially duplicates flow from conn & stream
                                            corr_id = self._makeCorrelationId(o_addr)
                                            if (
                                                this_correlation_id is not None
                                                and this_correlation_id != corr_id
                                            ):
                                                # likely a bug for this problem instance, but could happen in others
                                                print(
                                                    f"WARNING: #{idx} {symbol(addr)} tracks to different correlation ids: "
                                                    f"already set to {this_correlation_id}, but now resolved to {corr_id} for #{o_idx} {symbol(o_addr)}. Using the latest one"
                                                )
                                            this_correlation_id = corr_id
                                            break
                        if this_correlation_id is None:
                            print(
                                f"WARNING: #{idx} {symbol(addr)} don't have prior operations!"
                            )
                            this_correlation_id = self._makeCorrelationId(
                                addr
                            )  # using just some stray ID
                else:
                    this_event_track_id = MAIN_TRACK_UUID
                    if not assume_magic_trace:
                        this_correlation_id = self._makeCorrelationId(addr)

                    # below can be removed, as it only checks the match
                    if "buffs" in suppl:  # matching to its kernel launch
                        bufs = suppl["buffs"]
                        assert len(bufs) == 1  # individual ops use 1 buff only
                        bufs = bufs[0]
                        assert isinstance(bufs, tuple)

                        coll_has_reduce = collectiveHasReduction(suppl["coll_type"])
                        key_comp = (
                            (
                                suppl["dtype"],
                                suppl["coll_type"],
                                suppl["count"],
                                suppl["red_op"],
                            )
                            if coll_has_reduce
                            else (suppl["dtype"], suppl["coll_type"], suppl["count"])
                        )

                        match_found = False
                        for o_vals in call_list[call_idx + 1 :]:
                            o_idx, o_addr, _, _, _, o_suppl = o_vals
                            if o_addr == g_kernel_launch:
                                o_bufs = o_suppl["buffs"]
                                for b in o_bufs:
                                    if b == bufs:
                                        # testing if other call parameters match
                                        if coll_has_reduce:
                                            match_found = (
                                                o_suppl["dtype"],
                                                o_suppl["coll_type"],
                                                o_suppl["count"],
                                                o_suppl["red_op"],
                                            ) == key_comp
                                        else:
                                            match_found = (
                                                o_suppl["dtype"],
                                                o_suppl["coll_type"],
                                                o_suppl["count"],
                                            ) == key_comp
                                        if match_found:
                                            # o_idx shouldn't collide with comm ids
                                            # flow_ids.append(o_idx) # duplicates conn & stream flows
                                            break
                                if match_found:
                                    break
                        if not match_found:
                            print(
                                f"WARNING: #{idx} {symbol(addr)} is not matched to any kernel launch"
                            )

                # 2. Emit events
                packet = builder.add_packet()
                packet.timestamp = ts_start  # Start time in nanoseconds
                packet.trusted_packet_sequence_id = PACKET_SEQUENCE_ID
                packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN if ts_end > 0 else TrackEvent.TYPE_INSTANT
                # Associate with the track
                packet.track_event.track_uuid = this_event_track_id
                packet.track_event.name = _makeEventName(addr, suppl)
                if not assume_magic_trace:
                    packet.track_event.correlation_id = this_correlation_id

                # matching communicators for flow_ids
                if (link_by & self.LINK_BY_COMM) > 0:
                    v = suppl.get("newcomm")
                    if v is not None and v!=0:
                        flow_ids.append(v)
                    v = suppl.get("comm")
                    if v is not None and v!=0:
                        flow_ids.append(v)
                if (link_by & self.LINK_BY_STREAM) > 0:
                    v = suppl.get("stream")
                    if v is not None:
                        flow_ids.append(v)
                if (link_by & self.LINK_BY_DEVICE) > 0:
                    v = suppl.get("comm_cudaDev")
                    if v is not None:
                        flow_ids.append(v)
                # saving flow_ids
                for f in flow_ids:
                    packet.track_event.flow_ids.append(f)

                self._add_annotation(packet, "Log line number", idx)
                self._add_annotation(packet, suppl)
                if caller_addr is not None and caller_addr != "n/a":
                    self._add_annotation(
                        packet,
                        "Called from",
                        f"{symbol(caller_addr)} @ {symbols[caller_addr][1]}",
                    )

                if ts_end > 0:
                    packet = builder.add_packet()
                    packet.timestamp = ts_end  # End time in nanoseconds
                    packet.track_event.type = TrackEvent.TYPE_SLICE_END
                    packet.track_event.track_uuid = this_event_track_id
                    packet.trusted_packet_sequence_id = PACKET_SEQUENCE_ID


def supplElement2text(key: str, value):
    if key in ("comm", "newcomm", "stream"): # use default decimals for comm_cudaDev
        return "0x%x" % value
    if key == "buffs":
        bufs = [f"(0x{b[0]:x}, 0x{b[1]:x})" for b in value]
        if len(bufs) > 1:
            return "[" + ", ".join(bufs) + "]"
        return bufs[0]
    if key == "dtype" and isinstance(value, int):
        return DTYPE_NAMES[value]
    if key == "red_op" and isinstance(value, int):
        return RED_OP_NAMES[value]
    if key == "coll_type" and isinstance(value, int):
        return COLL_TYPES[value]

    return value


def print_per_thread(symbols: dict[str, tuple[str, str]], calls: dict[int, list[list]]):
    # finding the earliest timestamp
    min_ts = min(
        min(vals[2], vals[3] if vals[3] > 0 else vals[2])
        for clist in calls.values()
        for vals in clist
    )

    symbol = lambda x: symbols[x][0]  # noqa: E731

    print("Per thread calls begin ----->")
    print("log_line_num, func, ts_start_ns, ts_end_ns, duration_ns, supplement, caller")
    for tid, call_list in calls.items():
        print(f"Thread {tid}:")
        prev_ts_end = 0
        for vals in call_list:
            idx, addr, ts_start, ts_end, caller_addr, suppl = vals
            kernel_launch = addr == g_kernel_launch
            if not kernel_launch or ts_end > 0:  # kernel launches might
                # not have completions when a kernel hangs
                assert isNoExit(addr) or (0 < ts_end and ts_start < ts_end)
                assert kernel_launch or prev_ts_end < ts_start, f"prev_ts_end={prev_ts_end:x} < ts_start={ts_start:x} for #{idx} {symbol(addr)}"
                if not kernel_launch: # it's async, so it's end time doesn't count
                    prev_ts_end = max(ts_end, ts_start) # to cope with ts_end==0
            s = {
                k: supplElement2text(k, v)
                for k, v in suppl.items()
                if not k.startswith("_") and (kernel_launch or k != "coll_type")
            }
            print(
                f" #{idx}, {symbol(addr)}, {ts_start - min_ts}, {ts_end - min_ts if ts_end > 0 else 0}, {ts_end - ts_start if ts_end > 0 else 0}, {s}, {symbol(caller_addr)}"
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

    calls, symbols = parseLog(logfile)

    # report_single_overlaps(symbols, calls)
    print_per_thread(symbols, calls)

    print("address -> symbol mapping:")
    for addr, descr in symbols.items():
        print(f"{addr} := {descr[0]} @ {descr[1]}")

    MyTraceBuilder().make_trace(tracepath + "_comm_magic-trace.pftrace", symbols, calls, use_tid_pid=True, link_by=MyTraceBuilder.LINK_BY_COMM)
    MyTraceBuilder().make_trace(tracepath + "_stream_magic-trace.pftrace", symbols, calls, use_tid_pid=True, link_by=MyTraceBuilder.LINK_BY_STREAM)
    MyTraceBuilder().make_trace(tracepath + "_device_magic-trace.pftrace", symbols, calls, use_tid_pid=True, link_by=MyTraceBuilder.LINK_BY_DEVICE)

    #MyTraceBuilder().make_trace(tracepath + "_comm.pftrace", symbols, calls, use_tid_pid=False, link_by=MyTraceBuilder.LINK_BY_COMM)
    #MyTraceBuilder().make_trace(tracepath + "_stream.pftrace", symbols, calls, use_tid_pid=False, link_by=MyTraceBuilder.LINK_BY_STREAM)
    #MyTraceBuilder().make_trace(tracepath + "_device.pftrace", symbols, calls, use_tid_pid=False, link_by=MyTraceBuilder.LINK_BY_DEVICE)


def getFiles() -> tuple[str, str]:
    logfile = g_sym_filepath
    trace = g_trace_filepath
    if len(sys.argv) >= 2:
        logfile = sys.argv[1]
        if not os.path.isabs(logfile):
            logfile = g_work_dir + "/" + logfile

        root, _ = os.path.splitext(os.path.basename(logfile))
        trace = os.path.dirname(logfile) + f"/{root}"  # no extension here!

    return logfile, trace


if __name__ == "__main__":
    process_log(*getFiles())
