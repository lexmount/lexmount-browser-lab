# Shared Egress Proxy

This experiment utility makes a remote Lexmount browser and local Chrome use
the same evaluator-host egress. Start one loopback listener without credentials
for local Chrome and one authenticated listener behind a short-lived TCP tunnel
for Lexmount. The proxy denies private-network targets and permits only ports
80 and 443.

It is not a general-purpose proxy service. Keep the external listener
authenticated, use an ephemeral credential, and terminate the tunnel after the
paired run.

Pass the external listener credential with `--password-file` (a `0600` file),
not `--password`, so it is not visible through the process list.
