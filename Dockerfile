FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY analytics_mcp/ analytics_mcp/

RUN pip install --no-cache-dir .

ENV MCP_TRANSPORT=http

CMD ["analytics-mcp-http"]
