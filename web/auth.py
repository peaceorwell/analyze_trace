import os
import ssl


class AuthError(Exception):
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _ldap_imports():
    try:
        from ldap3 import ALL, Connection, Server, Tls
        from ldap3.core.exceptions import LDAPException
        from ldap3.utils.conv import escape_filter_chars
    except ImportError as e:
        raise AuthError("LDAP auth requires the ldap3 package. Run `uv sync --extra web`.") from e
    return ALL, Connection, Server, Tls, escape_filter_chars, LDAPException


def _ldap_server():
    ALL, _, Server, Tls, _, _ = _ldap_imports()
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
        "cn",
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


def _entry_user(entry, fallback_username: str = "") -> dict:
    display_attr = _env("LDAP_DISPLAY_NAME_ATTR", "displayName")
    mail_attr = _env("LDAP_MAIL_ATTR", "mail")
    username = (
        _value(entry, "sAMAccountName")
        or (_value(entry, "userPrincipalName").split("@", 1)[0] if _value(entry, "userPrincipalName") else "")
        or fallback_username
    )
    display_name = _value(entry, display_attr) or _value(entry, "cn") or username
    return {
        "username": username,
        "display_name": display_name,
        "email": _value(entry, mail_attr),
        "dn": getattr(entry, "entry_dn", ""),
    }


def search_users(query: str, limit: int = 8) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []

    _, Connection, _, _, escape_filter_chars, LDAPException = _ldap_imports()
    base_dn = _env("LDAP_BASE_DN")
    bind_dn = _env("LDAP_BIND_DN")
    bind_password = _env("LDAP_BIND_PASSWORD")
    if not base_dn or not bind_dn or not bind_password:
        return []

    limit = max(1, min(int(limit or 8), 20))
    safe = escape_filter_chars(query)
    contains = f"*{safe}*"
    prefix = f"{safe}*"
    search_filter = (
        "(|"
        f"(sAMAccountName={prefix})"
        f"(userPrincipalName={prefix})"
        f"(mail={prefix})"
        f"(displayName={contains})"
        f"(cn={contains})"
        ")"
    )

    server = _ldap_server()
    try:
        service = Connection(server, user=bind_dn, password=bind_password, auto_bind=True)
    except LDAPException as e:
        raise AuthError("LDAP service bind failed") from e
    try:
        try:
            service.search(base_dn, search_filter, attributes=_attrs(), size_limit=limit)
        except LDAPException as e:
            raise AuthError("LDAP user search failed") from e
        users = []
        for entry in list(service.entries)[:limit]:
            try:
                _require_group(entry)
            except AuthError:
                continue
            user = _entry_user(entry)
            if user.get("username"):
                users.append(user)
        return users
    finally:
        service.unbind()


def authenticate(username: str, password: str) -> dict:
    username = (username or "").strip()
    if not username or not password:
        raise AuthError("Username and password are required")

    _, Connection, _, _, escape_filter_chars, LDAPException = _ldap_imports()
    server = _ldap_server()
    dn_template = _env("LDAP_USER_DN_TEMPLATE")

    if dn_template:
        user_dn = dn_template.format(username=username)
        try:
            conn = Connection(server, user=user_dn, password=password, auto_bind=True)
        except LDAPException as e:
            raise AuthError("Invalid username or password") from e
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

    try:
        service = Connection(server, user=bind_dn, password=bind_password, auto_bind=True)
    except LDAPException as e:
        raise AuthError("LDAP service bind failed") from e
    try:
        try:
            found = service.search(base_dn, search_filter, attributes=_attrs(), size_limit=2)
        except LDAPException as e:
            raise AuthError("LDAP user search failed") from e
        if not found:
            raise AuthError("Invalid username or password")
        if len(service.entries) != 1:
            raise AuthError("LDAP user search did not return exactly one user")
        entry = service.entries[0]
        user_dn = entry.entry_dn
        _require_group(entry)
        user = _entry_user(entry, username)
        display_name = user["display_name"] or username
        email = user["email"]
    finally:
        service.unbind()

    try:
        user_conn = Connection(server, user=user_dn, password=password, auto_bind=True)
    except LDAPException as e:
        raise AuthError("Invalid username or password") from e
    user_conn.unbind()
    return {
        "username": username,
        "display_name": display_name,
        "email": email,
        "dn": user_dn,
    }
