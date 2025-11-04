import os

g_outfile = "rccl2.bt"

g_work_dir = os.path.dirname(__file__)
g_outfilepath = g_work_dir + "/" + g_outfile

g_lib = "/opt/rocm/lib/librccl.so.1"

# function names that doesn't need special handling.
# To fund which function might be needed in principle, make a test run with a wildcard probe
g_functions_simple = [
    "ncclGroupStart",
    "ncclGroupEnd",
    # "ncclGetVersion",
    # "ncclGetUniqueId",
    # "ncclCommCount",
]

# ncclAllGather
# ncclAllReduce
# ncclReduceScatter
# ncclCommInitRankConfig
# ncclCommSplit

g_config = """config = {
    // max_map_keys = 20;
    perf_rb_pages = 4096
}"""

g_tpl_uprobe = """{
    $thrd = tid(init);  // tid(init) is important for v0.23+
    $ip = reg("ip");

    $dpt = @depth[$thrd];
    @depth[$thrd]++;

    $ts_start = nsecs;
    if ($dpt <= 0){
        if ($dpt < 0){
            printf("NEGATIVE DEPTH");
            // signal("KILL");
            exit();
        }

        // @start[$thrd] = $ts_start;
        @funcs[$thrd] = $ip;

        %(custom)s
    }
}
"""

g_tpl_uretprobe = """{
    $ts_end = nsecs;
    $thrd = tid(init); // tid(init) is important for v0.23+

    @depth[$thrd]--;
    $dpt = @depth[$thrd];
    // $ts_start = @start[$thrd];
    $callee_addr = @funcs[$thrd];

    if ($dpt <= 0){
        if ($dpt < 0){
            printf("NEGATIVE DEPTH");
            // signal("KILL");
            exit();
        }

        // delete(@start, $thrd);
        // delete(@start[$thrd]);

        // delete(@funcs, $thrd);
        delete(@funcs[$thrd]);

        %(custom)s
    }
}
"""

def materialize_template() -> str:
    def make_header(probe, func):
        return f"{probe}:{g_lib}:{func}"

    func_AllRedScat = ["ncclAllReduce", "ncclReduceScatter"]
    uretprobe_funcs_simple = g_functions_simple + func_AllRedScat + ["ncclAllGather"]

    entry_pfx = r'printf("%s,%llx,%u,%llx'
    entry_vars = r'\n", probe, $ip, $thrd, $ts_start'

    exit_pfx = r'printf("%s,%llx,%llx,%u,%llx'
    exit_vars = r'\n", probe, reg("ip"), $callee_addr, $thrd, $ts_end'

    # note, parser assumes all values except for a thread id, to be in hex

    ret = [
        g_config,
        "\n////////////// SIMPLE FUNCTIONS /////////\n",
        ",\n".join([make_header("uprobe", f) for f in g_functions_simple]),
        g_tpl_uprobe % {"custom": entry_pfx + entry_vars + ");"},
        ",\n".join([make_header("uretprobe", f) for f in uretprobe_funcs_simple]),
        g_tpl_uretprobe % {"custom": exit_pfx + exit_vars + ");"},
        ###
        "////////////// SPECIAL FUNCTIONS /////////\n",
        ",\n".join([make_header("uprobe", f) for f in func_AllRedScat]),
        g_tpl_uprobe  # retprobe is "simple" above
        % {
            "custom": entry_pfx
            + (r",%llx,%x,%x,%llx" + entry_vars + ", arg2,arg3,arg4,arg5);")
        },
        ###
        make_header("uprobe", "ncclAllGather"),
        g_tpl_uprobe  # retprobe is "simple" above
        % {"custom": entry_pfx + r",%llx,%x,%llx" + entry_vars + ", arg2,arg3,arg4);"},
        ###
        make_header("uprobe", "ncclCommInitRankConfig"),
        g_tpl_uprobe
        % {
            "custom": "@comm[$thrd] = arg0;\n        "
            + (entry_pfx + r",%x,%x" + entry_vars + ", arg1,arg2);")
            # looks like a compiler is smart enough to not pass 128bytes of freaking by value ncclUniqueId in regs
        },
        make_header("uretprobe", "ncclCommInitRankConfig"),
        ",",
        make_header("uretprobe", "ncclCommSplit"),
        g_tpl_uretprobe
        % {
            "custom": (exit_pfx + r",%llx" + exit_vars + ", *(uint64*)@comm[$thrd]);")
            + "\n        delete(@comm[$thrd]);"
        },
        ###
        make_header("uprobe", "ncclCommSplit"),
        g_tpl_uprobe  # retprobe is the same as for ncclCommInitRankConfig
        % {
            "custom": "@comm[$thrd] = arg3;\n        "
            + (entry_pfx + r",%llx,%x,%x" + entry_vars + ", arg0,arg1,arg2);")
        },
    ]

    return "\n".join(ret)


def generate_bpftrace(outfile):
    bpft = materialize_template()

    with open(outfile, "w") as f:
        f.write(bpft)


if __name__ == "__main__":
    generate_bpftrace(g_outfilepath)
