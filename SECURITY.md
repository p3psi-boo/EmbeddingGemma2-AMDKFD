# Security and operational scope

This is an experimental single-user GPU backend. The launcher binds to loopback;
the HTTP server has no built-in authentication or TLS. Use an authenticated
reverse proxy with request-size, concurrency and timeout limits before exposing
it beyond the local machine. Do not treat GPU access or the process lock as a
multi-tenant security boundary.

Only load model files and kernel code objects from trusted sources. The model
download script pins a revision and checks SHA-256. GPU workloads exercise kernel
and firmware interfaces, and a fault can disrupt other GPU users through an
automatic driver reset.

For a sensitive report, use the repository hosting provider's private security
advisory channel when enabled. Do not put credentials, host inventories, memory
dumps or unredacted logs in a public issue. If private reporting is unavailable,
request a private reporting channel without including sensitive details.

No response-time SLA or production support commitment is currently offered.
