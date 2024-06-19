# List of Verilog reserved keywords to exclude from the results
verilog_keywords = [
    "always", "ifnone", "rpmos", "and", "rtran", "assign", "inout", "rtranif0",
    "begin", "input", "rtranif1", "buf", "integer", "scalared", "bufif0", "join", "small",
    "bufif1", "large", "case", "macromodule", "casex", "medium", "strong0",
    "casez", "module", "strong1", "cmos", "nand", "supply0", "deassign", "negedge", "supply1",
    "default", "nmos", "nor", "task", "disable", "not",
    "edge", "notif0", "tran", "else", "notif1", "tranif0", "end", "or", "tranif1",
    "endcase", "output", "tri", "endmodule", "parameter", "tri0", "endfunction", "pmos", "tri1",
    "endprimitive", "posedge", "triand", "primitive", "trior", "pull0", "trireg",
    "endtask", "pull1", "vectored", "pullup", "wait", "for", "pulldown", "wand",
    "force", "rcmos", "weak0", "forever", "real", "weak1", "fork", "while",
    "function", "reg", "wire", "highz0", "release", "wor", "highz1", "repeat", "xnor",
    "if", "rnmos", "xor", "automatic", "incdir", "pulsestyle_ondetect", "cell", "pulsestyle_onevent",
    "config", "instance", "signed", "endconfig", "liblist", "showcancelled", "endgenerate", "library", "unsigned",
    "generate", "localparam", "use", "genvar", "noshowcancelled", "accept_on", "export", "ref", "alias",
    "extends", "restrict", "always_comb", "extern", "return", "always_ff", "final", "s_always", "always_latch",
    "first_match", "s_eventually", "assert", "foreach", "s_nexttime", "assume", "forkjoin", "s_until", "before",
    "global", "s_until_with", "bind", "iff", "sequence", "bins", "ignore_bins", "shortint", "binsof",
    "illegal_bins",
    "shortreal", "bit", "implies", "solve", "break", "import", "static", "byte", "inside", "string", "chandle",
    "int", "strong", "checker", "interface", "struct", "class", "intersect", "super", "clocking", "join_any",
    "sync_accept_on", "const", "join_none", "sync_reject_on", "constraint", "let", "tagged", "context", "local",
    "this", "continue", "logic", "throughout", "cover", "longint", "timeprecision", "covergroup", "matches",
    "timeunit", "coverpoint", "modport", "type", "cross", "new", "typedef", "dist", "nexttime", "union", "do",
    "null", "unique", "endchecker", "package", "unique0", "endclass", "packed", "until", "endclocking", "priority",
    "until_with", "endgroup", "program", "untyped", "endinterface", "property", "var", "endpackage", "protected",
    "virtual", "endprogram", "pure", "void", "endproperty", "rand", "wait_order", "endsequence", "randc", "weak",
    "enum", "randcase", "wildcard", "eventually", "randsequence", "with", "expect", "reject_on", "within", "true", "false",
    "xor", "wire", "reg", "integer", "real", "for", "pragma", "in", "out", "#(", "none"  #, "ifdef", "endif", "elseif" possibly add back: conditional compilations
]


non_synthesizable_verilog_keywords = [
    r"->",
    r"\$display\s*\((?:.*?)\);?",  # System task for displaying
    r"\$readmemb\s*\((?:.*?)\);?",  # System task for reading binary data from a file
    r"\$readmemh\s*\((?:.*?)\);?",  # System task for reading hex data from a file
    r"\$write\b;?",  # System task for writing to files
    r"\$write\w\b;?",  # System task for writing binary data to a file
    #r'\$\w+\b',
    #r'\$\w+\(.*\)',    # Disable. this is too aggressive: catching $clog2
    r"\$monitor\s*\((?:.*?)\);?",  # System task for monitoring changes
    r"\$strobe\s*\((?:.*?)\);?",   # System task for strobing at the end of simulation time
    r"\$finish(?:\s*\((?:.*?)\))?;?",  # System task for finishing simulation
    r"\$finish;?",  # System task for finishing simulation
    r"\$stop(?:\s*\((?:.*?)\))?;?",    # System task for stopping simulation
    r"\$time\b;?",  # System function for current simulation time
    r"\#\d+'[sdbho]\d*;?",  # Time delays, more precisely defined,
    r"\#\d+\s*;?",  # Time delays without units
    r"\balways\s+@\(\s*(?:posedge|negedge)\s+\$\w+\);?",  # Always blocks sensitive to system task changes
    r"\$fopen\s*\((?:.*?)\);?",  # System function for opening files
    r"\$fwrite\s*\((?:.*?)\);?",  # System function for writing to files
    r"\$fread\s*\((?:.*?)\);?",   # System function for reading from files
    r"\$fclose\s*\((?:.*?)\);?",  # System function for closing files
    r"\$dumpfile\s*\((?:.*?)\);?",  # System task for specifying dump file
    r"\$dumpvars\s*\((?:.*?)\);?",  # System task for specifying variables to dump
    r"\$assert\s*\((?:.*?)\);?",  # System task for assertions in simulation
    r"\$random\b;?",  # System function for generating random numbers
    r"\$urandom\b;?", # System function for generating unsigned random numbers
    r"\$period\b;?",  # Matches `$period` system task
    r"\$realtime\b;?",  # System function for real simulation time
    r"\$timeformat\b;?",  # System task for setting the time format
     r'^`timescale\s+.*\n', # Directive for setting the time scale
    r"^.*\bspecparam\b.*$;?",
    r"defparam\s+\w+\.\w*\b(sim_|test_|mock_|fake_|dummy_|delay)\w*\b;?",
    r"defparam\s+\w+\.\w*\b(log_level|test_mode|test_vector)\b;?",
    r"defparam\s+\w+\.\w*\s+=\s+\w+\s*;.*//.*simulation;?",
    r"\$(?:setup|hold|width|recovery|removal)\b;?",  # Timing checks
    r"\$hold\s*\([^)]*\);?",
    r"\$sdf_annotate\b;?",  # SDF annotation for timing simulation
    r"\bcovergroup\b;?",  # Coverage group for functional coverage
    r"\$coverage\w*\b;?",  # Matches various coverage-related system tasks
    r"\$test\$plusargs\b;?",  # Checks for command-line plusargs
    r"\$value\$plusargs\b;?",  # Retrieves command-line plusargs values
    r"\$signed\b;?",  # System function for signed interpretation
    r"\$unsigned\b;?",  # System function for unsigned interpretation
    r"\$cast\b;?",  # System function for type casting
    r"\$size\b;?",  # System function for determining the size of a variable
    r"\$fatal\b;?",  # System task for triggering a fatal error
    r"\$error\b;?",  # System task for reporting an error
    r"\$warning\b;?",  # System task for issuing a warning
    r"\$info\b;?",  # System task for providing information
    r"\b(?:initial|always)\b.*?`ifdef\s+SIMULATION;?",  # Conditional compilation for simulation
    r"\$system\b;?",  # System task for executing system commands
    r"\$exit\b;?",  # System function to exit the simulation
    r"\$countones\b;?",  # System function for counting ones
    r"\$onehot\b;?",  # System function for checking one-hot
    r"\$onehot0\b;?",  # System function for checking one-hot or zero
    r"\$isunknown\b;?",  # System function for checking unknown state
    r"\$sampled\b;?",  # System task for sampled value of a variable
    r"\$past\b;?",  # System function for past value of a signal (SVA)
    r"\$rose\b;?",  # System function for detecting rising edge (SVA)
    r"\$fell\b;?",  # System function for detecting falling edge (SVA)
    r"\$stable\b;?",  # System function for detecting stability (SVA)
    r"\$changed\b;?",  # System function for detecting change (SVA)
    r"^\s*#\s*directive;?",
    r'^#.*;?', # Remove lines that start with #: non-synth
    # Time delays without units
]

non_synthesizable_verilog_strings = [
    "$display",  # System task for displaying
    "$monitor",  # System task for monitoring changes
    "$strobe",   # System task for strobing at the end of simulation time
    "$finish",   # System task for finishing simulation
    "$stop",     # System task for stopping simulation
    "$time",     # System function for current simulation time
    "#",         # Time delays, more precisely defined
    "$fopen",    # System function for opening files
    "$fwrite",   # System function for writing to files
    "$fread",    # System function for reading from files
    "$fclose",   # System function for closing files
    "$dumpfile", # System task for specifying dump file
    "$dumpvars", # System task for specifying variables to dump
    "$assert",   # System task for assertions in simulation
    "$random",   # System function for generating random numbers
    "$urandom",  # System function for generating unsigned random numbers
    "$display",  # Matches `$display` without arguments or optional semicolon
    "$period;?",   # Matches `$period` system task
    "$realtime", # System function for real simulation time
    "$timeformat", # System task for setting the time format
    "`timescale",  # Directive for setting the time scale
    "specparam", "defparam",  # Matches 'specparam' and 'defparam' for special parameters
    "$setup", "$hold", "$width", "$recovery", "$removal",  # Timing checks
    "$sdf_annotate",  # SDF annotation for timing simulation
    "covergroup",     # Coverage group for functional coverage
    "$coverage",      # Matches various coverage-related system tasks
    "$test$plusargs", # Checks for command-line plusargs
    "$value$plusargs",# Retrieves command-line plusargs values
    "$signed",        # System function for signed interpretation
    "$unsigned",      # System function for unsigned interpretation
    "$cast",          # System function for type casting
    "$size",          # System function for determining the size of a variable
    #"$clog2",         # System function for calculating the ceiling log base 2
    "$fatal",         # System task for triggering a fatal error
    "$error",         # System task for reporting an error
    "$warning",       # System task for issuing a warning
    "$info",          # System task for providing information
    "`ifdef SIMULATION", # Conditional compilation for simulation
    "$system",        # System task for executing system commands
    "$exit",          # System function to exit the simulation
    "$countones",     # System function for counting ones
    "$onehot",        # System function for checking one-hot
    "$onehot0",       # System function for checking one-hot or zero
    "$isunknown",     # System function for checking unknown state
    "$sampled",       # System task for sampled value of a variable
    "$past",          # System function for past value of a signal (SVA)
    "$rose",          # System function for detecting rising edge (SVA)
    "$fell",          # System function for detecting falling edge (SVA)
    "$stable",        # System function for detecting stability (SVA)
    "$changed",       # System function for detecting change (SVA)
]

non_synth_path_keywords = [
    "testbench", "tb",
    "test", "tests",
    "sim", "simulation",
    "pragma",
    "COQ",
    "formal",
    #"doc", "documentation",
    "examples", "demo",
    "synth_ignore", "nosynth",
    #"sv", "systemverilog", # Consider removing systemverilog
    "constraint", "sdc",
    #"utility", "utils",
    "verify", "verification"
]

COQ_keywords = [
    "Proof",
    "Proof.",
    "Proof .",
    "Qed",
    "Qed.",
    "Qed .",
    "Defined",
    "Defined.",
    "Defined .",
    "Compute",
    "Admitted",
    "Definition",
    "Lemma ",
    "Theorem ",
    "Corollary ",
    "Proposition ",
    "Inductive ",
    "CoInductive ",
    "Fixpoint ",
    "CoFixpoint ",
    "cofix",
    "fix",
    "Require Import ",
    "Require Export ",
    "Functional Scheme",
    "Declare scope",
    "Delimit scope",
    "Bind scope",
    "Existing class",
    ":=",
    '\\->',
    '\\/',
    '/\\',
    "Check",
    "forall",
    "<->",
    "->",
    "<>",
    "=>",
    "Local Arguments",
    "Set Implicit Arguments",
    "decide equality",
    "destruct",
    "lia",
    "rewrite",
]

single_line_removal = ["Open Scope", "Include", "include", "require import", "Require import", "pragma", "#TCQ", r'\bforce\b', r'\brelease\b']

# Markers for the beginning of multi-line blocks of invalid/non-verilog code
multiline_begin_words = [
    r"\/*\**\b",
    r"pragma protect data_block",
    r"Section Lemmas",
    #r"`ifdef",
    r"(*\**\b",
    r"Proof",
    r"Definition"
    r"Theorem",
    r"Lemma",
    r"Example",
    r"// Simulation Start",
    r"// Assertions Start",
    r"/* Simulation Code */",
    r"/* Begin Assertions */",
    r"specify"
]

# Markers for the end of multi-line blocks of invalid/non-verilog code
multiline_end_words = [
    r"\*+/+\b",
    #r"`endif",
    r"\bend\b",
    r"\bend_protected\b",
    r"\**)*\b",
    r"Qed",
    r"End Lemmas",
    r"Defined\s*\.?",
    r"Admitted",
    r"// Simulation End",
    r"// Assertions End",
    r"/* End Simulation Code */",
    r"/* End Assertions */",
    r"endspecify",
    r"pragma protect end"
]

directive_list = [
    "`ifdef",
    "`ifndef",
    "`elsif",
    "`else",
    "`endif",
    "`define",
    "`undef",
    "`include",
    "`line",
    "`timescale",
    "`default_nettype",
    "`celldefine",
    "`endcelldefine",
    "`protect",
    "`endprotect",
    "`protected",
    "`endprotected",
    "`pragma",
    "`resetall",
    "`unconnected_drive",
    "`nounconnected_drive",
    "`timeskew",
    "`default_decay_time",
    "`default_trireg_strength"
]
meaningful_identifiers = [
    r"(?<!\w)\d+\'d\d+(?!\w)",
    r"\blut\b",
    r"\breg\b", r"\bclk\b", r"\bin\b", r"\bout\b",
    r"\brng\b", r"\brnd\b", r"\brand\b",
    r"\brst\b", r"\breset\b",
    r"\benable\b", r"\ben\b",
    r"\bdata_in\b", r"\bdin\b",
    r"\bdata_out\b", r"\bdout\b",
    r"\baddr\b", r"\baddress\b",
    r"\bsel\b", r"\bselect\b",
    r"\bwr\b", r"\bwrite\b",
    r"\brd\b", r"\bread\b",
    r"\back\b", r"\backnowledge\b",
    r"\breq\b", r"\brequest\b",
    r"\birq\b", r"\binterrupt\b",
    r"\bvalid\b",
    r"\bready\b",
    r"\bpos\b", r"\bneg\b",
    r"\bsync\b", r"\basync\b",
    r"\bpulse\b",
    r"\bflag\b",
    r"\bbit\b", r"\bbyte\b", r"\bword\b",
    r"\bcounter\b",
    r"\bregister\b", r"\breg\b",
    r"\bbuffer\b",
    r"\bmux\b", r"\bdemux\b",
    r"\bencoder\b", r"\bdecoder\b",
    r"\bcomparator\b",
    r"\balu\b",
    r"\bsum\b",
    r"\bfsm\b", r"\bstate\b", r"\bnext_state\b",
    r"\bdelay\b",
    r"\btimeout\b",
    r"\bperiod\b",
    r"\bdata_signal\b",
    r"\brx_data\b",
    r"\btx_data\b",
    r"\baudio_sample\b",
    r"\bpixel_value\b",
    r"\bsensor_reading\b",
    r"\btemp_data\b",
    r"\bcontrol_flags\b",
    r"\bstatus_flags\b",
    r"\bcommand_reg\b",
    r"\bconfig_reg\b",
    r"\berror_code\b",
    r"\bmessage_length\b",
    r"\bpacket_buffer\b",
    r"\bstream_id\b",
    r"\bchannel_freq\b",
    r"\bvoltage_level\b",
    r"\bcurrent_limit\b",
    r"\bpower_mode\b",
    r"\bencryption_key\b",
    r"\bchecksum_value\b",
    r"\bframe_counter\b",
    r"\bevent_timer\b",
    r"\bqueue_depth\b",
    r"\buser_input\b",
    r"\bdisplay_buffer\b",
    r"\bread_pointer\b",
    r"\bwrite_pointer\b",
    r"\bmemory\b"
]


bad_words_lists = [multiline_begin_words, multiline_end_words, non_synthesizable_verilog_strings, COQ_keywords, single_line_removal]
reserved_words = verilog_keywords + non_synthesizable_verilog_keywords + non_synth_path_keywords + COQ_keywords + single_line_removal \
    + directive_list + multiline_begin_words + multiline_end_words + non_synthesizable_verilog_strings

def flatten_bad_words(bad_words_lists):
    temp = []
    for word_list in bad_words_lists:
        for word in word_list:
            if word not in temp:
                temp.append(word)
    return temp

bad_words = flatten_bad_words(bad_words_lists)

