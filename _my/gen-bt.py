import os

# every weird and inconsistent thing here is b/c the need to reduce amount of info dumped

g_outfile = "rccl2.1.bt"

g_work_dir = os.path.dirname(__file__)
g_outfilepath = g_work_dir + "/" + g_outfile

g_lib = "/opt/rocm/lib/librccl.so.1"

g_config = """config = {
    max_map_keys = 8;            // default max number of keys in a map
    unstable_map_decl=enable;    // for toplevel let @probes
    print_maps_on_exit = false;  // no useful info there
    perf_rb_pages = 4096      // size of output buffer in pages
    // perf_rb_pages = 1       // playing with this helps to cause or avoid the hang
}

// probes that don't have a dedicated body will use this map to translate a probe name to id.
// Not sure it's very efficient, but the alternative is to make a separate body per probe, which is
// to cumbersome. The map, though, mostly works.
let @probes = hash(9);  // number must correspond to the number of mappings in the begin{} block
begin{
    // uprobe, capital letters
    @probes["uprobe:/opt/rocm/lib/librccl.so.1:ncclGroupStart"] = "GS";
    @probes["uprobe:/opt/rocm/lib/librccl.so.1:ncclGroupEnd"] = "GE";

    // uretprobe, lowercase letters. Otherwise must correspond to uprobe
    @probes["uretprobe:/opt/rocm/lib/librccl.so.1:ncclGroupStart"] = "gs";
    @probes["uretprobe:/opt/rocm/lib/librccl.so.1:ncclGroupEnd"] = "ge";
    @probes["uretprobe:/opt/rocm/lib/librccl.so.1:ncclAllReduce"] = "ar";
    @probes["uretprobe:/opt/rocm/lib/librccl.so.1:ncclReduceScatter"] = "rs";
    @probes["uretprobe:/opt/rocm/lib/librccl.so.1:ncclAllGather"] = "ag";

    @probes["uretprobe:/opt/rocm/lib/librccl.so.1:ncclCommInitRankConfig"] = "ic";
    @probes["uretprobe:/opt/rocm/lib/librccl.so.1:ncclCommSplit"] = "cs";
}
"""


# function names that doesn't need special handling.
# To fund which function might be needed in principle, make a test run with a wildcard probe
g_functions_simple = [
    "ncclGroupStart",
    "ncclGroupEnd",
    # "ncclGetVersion",
    # "ncclGetUniqueId",
    # "ncclCommCount",
]


g_tpl_uprobe = """{
    $thrd = tid(init);  // tid(init) is important for v0.23+
    // $ip = reg("ip");

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
        // @funcs[$thrd] = $ip;

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
    // $callee_addr = @funcs[$thrd];

    if ($dpt <= 0){
        if ($dpt < 0){
            printf("NEGATIVE DEPTH");
            // signal("KILL");
            exit();
        }

        // delete(@start, $thrd);
        // delete(@start[$thrd]);

        // delete(@funcs, $thrd);
        // delete(@funcs[$thrd]);

        %(custom)s
    }
}
"""

g_tpl_LaunchKernel = r"""{
    // arg0 is void* kernelFn and is the same for all invocations
    // arg1 is bool is_ext and  is the same for all invocations
    $ts_start = nsecs;

    $n_tasks_m_1 = (arg2 & 3);
    $ptr = (uint64*) (arg2 & 0xFFFFFFFFFFFFFFFC);

    if (0==$n_tasks_m_1){
        printf("L %llx,%llx,%llx,%llx,%llx,%llx\n", tid(init),$ts_start,arg2,*$ptr,
            *($ptr+1),*($ptr+2) );
    }else if (1==$n_tasks_m_1){
        printf("L %llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx\n", tid(init),$ts_start,arg2,*$ptr,
            *($ptr+1),*($ptr+2),
            *($ptr+3),*($ptr+4)
        );
    }else if (2==$n_tasks_m_1){
        printf("L %llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx\n", tid(init),$ts_start,arg2,*$ptr,
            *($ptr+1),*($ptr+2),
            *($ptr+3),*($ptr+4),
            *($ptr+5),*($ptr+6)
        );
    }else if (3==$n_tasks_m_1){
        printf("L %llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx\n", tid(init),$ts_start,arg2,*$ptr,
            *($ptr+1),*($ptr+2),
            *($ptr+3),*($ptr+4),
            *($ptr+5),*($ptr+6),
            *($ptr+7),*($ptr+8)
        );
    }else{
        printf("UNSUPPORTED TASK COUNT = %u!\n", $n_tasks_m_1);
        exit();
    }
}
"""

g_tpl_kernelCallback = r"""{
    printf("l %llx,%llx\n", nsecs, arg0); // tid makes no sense here, it's HIP internal thread
}
"""


def materialize_template() -> str:
    def make_header(probe, func):
        return f"{probe}:{g_lib}:{func}"

    func_AllRedScat = ["ncclAllReduce", "ncclReduceScatter"]
    uretprobe_funcs_simple = g_functions_simple + func_AllRedScat + ["ncclAllGather"]

    # this is a generic header applicable to all probes, except kernel launch callback, which doesn't have tid
    entry_pfx = r'printf("%s %llx,%llx'
    assert entry_pfx.count("%s") == 1  # we do some substitues below

    def c_entry_pfx(id: str) -> str:
        return entry_pfx.replace("%s", id)

    entry_vars = r'\n", @probes[probe],$thrd,$ts_start'
    assert entry_vars.count("@probes[probe],") == 1  # same
    c_entry_vars = entry_vars.replace("@probes[probe],", "")

    exit_pfx = r'printf("%s %llx,%llx'
    assert exit_pfx.count("%s") == 1  # we do some substitues below

    def c_exit_pfx(id: str) -> str:
        return exit_pfx.replace("%s", id)

    exit_vars = r'\n", @probes[probe],$thrd, $ts_end'
    assert exit_vars.count("@probes[probe],") == 1  # same
    c_exit_vars = exit_vars.replace("@probes[probe],", "")

    def EntryWArgs(
        id: str, argnums: int | list | tuple | range, *, pfx: str = "", sfx: str = ""
    ) -> str:
        if isinstance(argnums, int):
            argnums = range(argnums)
        assert isinstance(argnums, (list, tuple, range))
        fmt = "".join(",%llx" for _ in argnums)
        vars = "".join(f",arg{i}" for i in argnums)
        return g_tpl_uprobe % {
            "custom": pfx + c_entry_pfx(id) + fmt + c_entry_vars + vars + ");" + sfx
        }

    ret = [
        g_config,
        "\n////////////// SIMPLE FUNCTIONS /////////\n",
        ",\n".join([make_header("uprobe", f) for f in g_functions_simple]),
        g_tpl_uprobe % {"custom": entry_pfx + entry_vars + ");"},
        ",\n".join([make_header("uretprobe", f) for f in uretprobe_funcs_simple]),
        g_tpl_uretprobe % {"custom": exit_pfx + exit_vars + ");"},
        ###
        "////////////// SPECIAL FUNCTIONS /////////\n",
        make_header("uprobe", "ncclAllReduce"),
        EntryWArgs("AR", 6),  # retprobe is "simple" above
        make_header("uprobe", "ncclReduceScatter"),
        EntryWArgs("RS", 6),  # retprobe is "simple" above
        ###
        make_header("uprobe", "ncclAllGather"),
        EntryWArgs("AG", 5),  # retprobe is "simple" above
        ###
        make_header("uprobe", "ncclCommInitRankConfig"),
        # looks like a compiler is smart enough to not pass 128bytes of freaking by value ncclUniqueId in regs
        EntryWArgs("IC", (1, 2), pfx="@comm[$thrd] = arg0;\n        "),
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
        # retprobe is the same as for ncclCommInitRankConfig
        EntryWArgs("CS", 3, pfx="@comm[$thrd] = arg3;\n        "),
        ############## TOTALLY CUSTOMIZED PROBES
        make_header("uprobe", "arechLaunchKernel"),
        g_tpl_LaunchKernel,
        make_header("uprobe", "arechCallback"),
        g_tpl_kernelCallback,
    ]

    return "\n".join(ret)


def generate_bpftrace(outfile):
    bpft = materialize_template()

    with open(outfile, "w") as f:
        f.write(bpft)


if __name__ == "__main__":
    generate_bpftrace(g_outfilepath)
