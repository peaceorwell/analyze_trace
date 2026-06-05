import os
import ssl


class AuthError(Exception):
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _ldap_imports():
    try:
        from ldap3 import ALL, Connection, Server, Tls
        from ldap3.utils.conv import escape_filter_chars
    except ImportError as e:
        raise AuthError("LDAP auth requires the ldap3 package. Run `uv sync --extra web`.") from e
    return ALL, Connection, Server, Tls, escape_filter_chars


def _ldap_server():
    ALL, _, Server, Tls, _ = _ldap_imports()
    url = _env("LDAP_URL")
    if not url:
        raise AuthError("LDAP_URL is required")

    ca_file = _env("LDAP_TLS_CA_FILE")
    tls = None
    if ca_file:
        tls = Tls(validate=ssl.CERT_REQUIRED, ca_certs_file=ca_file)
    return Server(url, get_info=ALL, tls=tls)


def _attrs():
    attrs = {
        _env("LDAP_DISPLAY_NAME_ATTR", "displayName"),
        _env("LDAP_MAIL_ATTR", "mail"),
        "memberOf",
        "sAMAccountName",
        "userPrincipalName",
    }
    return [attr for attr in attrs if attr]


def _value(entry, attr):
    if not attr or not hasattr(entry, attr):
        return ""
    value = getattr(entry, attr).value
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def _member_values(entry):
    if not hasattr(entry, "memberOf"):
        return []
    values = entry.memberOf.values
    if isinstance(values, list):
        return values
    return [values] if values else []


def _require_group(entry):
    required = _env("LDAP_REQUIRE_GROUP_DN")
    if not required:
        return
    members = {str(value).lower() for value in _member_values(entry)}
    if required.lower() not in members:
        raise AuthError("User is not in the required LDAP group")


def authenticate(username: str, password: str) -> dict:
    username = (username or "").strip()
    if not username or not password:
        raise AuthError("Username and password are required")

    _, Connection, _, _, escape_filter_chars = _ldap_imports()
    server = _ldap_server()
    dn_template = _env("LDAP_USER_DN_TEMPLATE")

    if dn_template:
        user_dn = dn_template.format(username=username)
        conn = Connection(server, user=user_dn, password=password, auto_bind=True)
        conn.unbind()
        return {"username": username, "display_name": username, "email": "", "dn": user_dn}

    base_dn = _env("LDAP_BASE_DN")
    bind_dn = _env("LDAP_BIND_DN")
    bind_password = _env("LDAP_BIND_PASSWORD")
    if not base_dn or not bind_dn or not bind_password:
        raise AuthError("LDAP_BASE_DN, LDAP_BIND_DN and LDAP_BIND_PASSWORD are required")

    user_filter = _env("LDAP_USER_FILTER", "(sAMAccountName={username})")
    safe_username = escape_filter_chars(username)
    search_filter = user_filter.format(username=safe_username)

    service = Connection(server, user=bind_dn, password=bind_password, auto_bind=True)
    try:
        if not service.search(base_dn, search_filter, attributes=_attrs(), size_limit=2):
            raise AuthError("Invalid username or password")
        if len(service.entries) != 1:
            raise AuthError("LDAP user search did not return exactly one user")
        entry = service.entries[0]
        user_dn = entry.entry_dn
        _require_group(entry)
        display_attr = _env("LDAP_DISPLAY_NAME_ATTR", "displayName")
        mail_attr = _env("LDAP_MAIL_ATTR", "mail")
        display_name = _value(entry, display_attr) or username
        email = _value(entry, mail_attr)
    finally:
        service.unbind()

    user_conn = Connection(server, user=user_dn, password=password, auto_bind=True)
    user_conn.unbind()
    return {
        "username": username,
        "display_name": display_name,
        "email": email,
        "dn": user_dn,
    }
