# Shared Egress Proxy

This experiment utility makes a remote Lexmount browser and local Chrome use
the same egress. Start one loopback listener without credentials for local
Chrome and configure Lexmount with an authenticated external proxy. The
loopback listener can chain to that same external proxy, so Chrome never needs
proxy credentials in its command line. The proxy denies private-network targets
and permits only ports 80 and 443.

It is not a general-purpose proxy service. Keep the external listener
authenticated, use an ephemeral credential, and terminate the tunnel after the
paired run.

Pass the external listener credential with `--password-file` (a `0600` file),
not `--password`, so it is not visible through the process list.

For an explicit shared upstream proxy, use the credential-free local listener
with `--upstream-proxy-server`, `--upstream-proxy-username`, and
`--upstream-proxy-password-file`. The upstream server URL must not embed
credentials. This mode is intended for a short-lived, authenticated proxy only;
do not expose the loopback listener beyond the evaluator host.

If cpolar emits its endpoint inside an escaped JSON log line, extract it with
`extract_cpolar_tcp_endpoint.py` rather than a broad shell regex. The helper
validates `host:port` and removes the JSON escape suffix before the value is
used for a Lexmount external proxy.
