from opensearchpy import OpenSearch

def create_client(
    host: str,
    port: int,
    user: str,
    password: str,
    ca_certs: str,
):
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, password),
        use_ssl=True,
        verify_certs=True,
        ca_certs=ca_certs,
    )
