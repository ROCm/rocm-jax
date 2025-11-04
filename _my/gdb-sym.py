import os
import os.path
import re
import sys

import gdb

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


def parse_log(logfile) -> tuple[list[str], dict[int, list[list]]]:
    """parses log into a unique set of addresses and a dict of per-thread calls"""
    if not os.path.exists(logfile):
        raise RuntimeError(f"logfile {logfile} not found")

    r_entry = re.compile(r"^\s*(\d+),\s*(0x[0-9a-fA-F]+),\s*(\d+),\s*(\d+)")
    r_exit = re.compile(
        r"^\s*\>(\d+),\s*(0x[0-9a-fA-F]+),\s*(0x[0-9a-fA-F]+),\s*(\d+),\s*(\d+),\s*(\d+)"
    )

    addresses: list[str] = []
    # maps tid -> [ [start_idx, addr, ts_start, ts_end, caller_addr] ]
    calls: dict[int, list[list]] = {}

    with open(logfile, "r") as file:
        for idx, line in enumerate(file):
            idx += 1  # zero idx adjust
            if m_entry := r_entry.match(line):
                addr = m_entry.group(2)
                assert addr and addr != "0"
                addresses.append(addr)

                tid = int(m_entry.group(3))
                assert tid > 0
                depth = int(m_entry.group(1))
                assert depth >= 0
                ts_start = int(m_entry.group(4))
                assert ts_start > 0

                if tid in calls:
                    _, prev_addr, prev_ts_start, prev_ts_end, prev_caller = calls[tid][
                        -1
                    ]
                    assert prev_ts_start < ts_start

                if depth <= 0:
                    assert tid not in calls or (
                        0 < prev_ts_end and prev_caller is not None
                    )
                    calls.setdefault(tid, []).append([idx, addr, ts_start, 0, None])
                else:
                    assert tid in calls
                    assert prev_addr != addr
                    # ^^ could fail for recursions, but there should be none of them

            elif m_exit := r_exit.match(line):
                caller_addr = m_exit.group(2)
                assert caller_addr and caller_addr != "0"
                callee_addr = m_exit.group(3)
                assert callee_addr and callee_addr != "0", (
                    "The log is broken, callee address isn't set"
                )
                addresses.append(caller_addr)
                addresses.append(callee_addr)

                depth = int(m_exit.group(1))
                assert depth >= 0
                tid = int(m_exit.group(4))
                assert tid > 0
                ts_end = int(m_exit.group(5))
                assert ts_end > 0
                elapsed = int(m_exit.group(6))
                assert elapsed > 0

                prev_idx, addr, ts_start, old_ts_end, old_caller = calls[tid][-1]
                assert old_ts_end == 0 and old_caller is None
                assert ts_start < ts_end
                assert elapsed != ts_end, "elapsed == ts_end, the log is broken"
                assert ts_start + elapsed == ts_end, (
                    f"Line#{idx}, ts_end={ts_end}, elapsed={elapsed}, ts_start={ts_start}"
                )
                assert callee_addr == addr

                if depth <= 0:
                    calls[tid][-1][3] = ts_end  # marking as not running
                    calls[tid][-1][4] = caller_addr

            elif line:
                raise RuntimeError(f"Unparsable line#{idx} {line}")

    return sorted(list(frozenset(addresses))), calls


def get_symbols(addresses: list[str]) -> dict[str, tuple[str, str]]:
    """Obtain symbols for given set of addresses"""
    r_sym = re.compile(r"^(.+)\s+in section (.*)$")
    ret: dict[str, tuple[str, str]] = {}
    for addr in addresses:
        resp = gdb.execute(f"info symbol 0x{addr}", to_string=True)
        if not isinstance(resp, str):
            raise RuntimeError(f"For '{addr}' GDB returned '{resp}'")
        if m := r_sym.match(resp):
            assert addr not in ret
            ret[addr] = (m.group(1), m.group(2))
        else:
            raise RuntimeError(f"For '{addr}' can't match GDB responce '{resp}'")
    return ret


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


def make_trace_packets(
    builder: TraceProtoBuilder,
    symbols: dict[str, tuple[str, str]],
    calls: dict[int, list[list]],
):
    symbol = lambda x: symbols[x][0]  # noqa: E731

    def add_annotation(packet, name, value):
        annotation = packet.track_event.debug_annotations.add()
        assert isinstance(name, str)
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
            idx, addr, ts_start, ts_end, caller_addr = vals

            # 2. Emit events for this custom track
            packet = builder.add_packet()
            packet.timestamp = ts_start  # Start time in nanoseconds
            packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
            # Associate with the track
            packet.track_event.track_uuid = CUSTOM_TRACK_UUID
            packet.track_event.name = f"#{idx} {symbol(addr)}"
            # packet.track_event.name = symbol(addr)
            packet.trusted_packet_sequence_id = PACKET_SEQUENCE_ID

            add_annotation(packet, "Log line number", idx)
            add_annotation(
                packet,
                "Called from",
                f"{symbol(caller_addr)} @ {symbols[caller_addr][1]}",
            )

            packet = builder.add_packet()
            packet.timestamp = ts_end  # End time in nanoseconds
            packet.track_event.type = TrackEvent.TYPE_SLICE_END
            packet.track_event.track_uuid = CUSTOM_TRACK_UUID
            packet.trusted_packet_sequence_id = PACKET_SEQUENCE_ID


def make_trace(
    tracepath: str, symbols: dict[str, tuple[str, str]], calls: dict[int, list[list]]
):
    builder = TraceProtoBuilder()
    make_trace_packets(builder, symbols, calls)

    with open(tracepath, "wb") as f:
        f.write(builder.serialize())

    print(f"Trace written to {tracepath}")
    print("Open with https://magic-trace.org/ or https://ui.perfetto.dev")


def process_log(logfile, tracepath):
    print(f"Using log file {logfile}")

    addresses, calls = parse_log(logfile)
    symbols = get_symbols(addresses)

    report_single_overlaps(symbols, calls)

    symbol = lambda x: symbols[x][0]  # noqa: E731

    print("Per thread calls begin ----->")
    print("log_num, func, ts_start_ns, ts_end_ns, duration_ns, caller")
    for tid, call_list in calls.items():
        print(f"Thread {tid}:")
        prev_ts_end = 0
        for vals in call_list:
            idx, addr, ts_start, ts_end, caller_addr = vals
            assert 0 < ts_end
            assert prev_ts_end < ts_start
            assert ts_start < ts_end
            prev_ts_end = ts_end
            print(
                f" #{idx}, {symbol(addr)}, {ts_start}, {ts_end}, {ts_end - ts_start}, {symbol(caller_addr)}"
            )
    print("-----> end of per thread calls")

    print("address -> symbol mapping:")
    for addr, descr in symbols.items():
        print(f"{addr} := {descr[0]} @ {descr[1]}")

    make_trace(tracepath, symbols, calls)


if __name__ == "__main__":
    process_log(g_sym_filepath if len(sys.argv)<2 else sys.argv[1], g_trace_filepath)
