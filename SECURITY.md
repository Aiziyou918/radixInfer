# Security Policy

## Reporting a Vulnerability

If you believe you have found a security vulnerability in `radixInfer`, please do not open a public issue with exploit details.

Instead, report it privately to the project maintainers through a private channel you control. Include:

- a clear description of the issue
- affected versions or commit ranges when known
- reproduction steps or a minimal proof of concept
- potential impact
- any suggested mitigation

Please avoid public disclosure until the issue has been reviewed and a fix or mitigation plan is available.

## Scope

Security-relevant areas may include, but are not limited to:

- HTTP request handling
- process orchestration
- queue / ZMQ message boundaries
- model loading paths
- filesystem or shell interaction introduced by tooling

## Response Expectations

The project aims to acknowledge reports promptly and evaluate impact before public disclosure, but no fixed SLA is guaranteed.
