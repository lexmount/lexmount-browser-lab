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

If cpolar emits its endpoint inside an escaped JSON log line, extract it with
`extract_cpolar_tcp_endpoint.py` rather than a broad shell regex. The helper
validates `host:port` and removes the JSON escape suffix before the value is
used for a Lexmount external proxy.
