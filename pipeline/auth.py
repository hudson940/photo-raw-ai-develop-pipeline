"""Operator authentication via Keycloak (in-app login form) + stateless sessions.

Flow (answer to "in-app login form"):
  1. The SPA posts {username, password} to /auth/login.
  2. We exchange them at Keycloak's token endpoint (OAuth2 password/direct grant).
  3. On success we read the returned access token's claims (username + realm roles)
     and mint our OWN signed session cookie (HMAC-SHA256). That cookie — not the
     Keycloak token — is what every later request presents. Keycloak stays the
     user store / password authority; sessions are stateless so any webui replica
     validates them without shared state.

Roles: 'super_admin' (manage users, see all photos) and 'editor' (own photos only).

The Users panel calls Keycloak's Admin REST API with a service-account token
(client-credentials grant on the same confidential client, which must hold the
realm-management roles manage-users / view-users / query-users).

If KEYCLOAK_URL is unset the whole module reports disabled and the web layer
falls back to the single-admin password (or open dev mode).
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import CONFIG

log = logging.getLogger("auth")

SUPER_ADMIN = "super_admin"
EDITOR = "editor"
_VALID_ROLES = (SUPER_ADMIN, EDITOR)

# stable secret if configured, else random per-process (sessions drop on restart)
_SESSION_SECRET = (CONFIG.session_secret or secrets.token_hex(32)).encode()


class AuthError(Exception):
    """Login / Keycloak interaction failure with a safe, user-facing message."""


def enabled() -> bool:
    return bool(CONFIG.keycloak_url)


# ---------------------------------------------------------------- HTTP helpers
def _realm_base() -> str:
    return f"{CONFIG.keycloak_url}/realms/{urllib.parse.quote(CONFIG.keycloak_realm)}"


def _admin_base() -> str:
    return f"{CONFIG.keycloak_url}/admin/realms/{urllib.parse.quote(CONFIG.keycloak_realm)}"


def _http(url: str, *, data=None, method="GET", token=None, form=False, timeout=15):
    headers = {"Accept": "application/json"}
    body = None
    if data is not None:
        if form:
            body = urllib.parse.urlencode(data).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"error": raw.decode("utf-8", "replace")[:200]}
        return exc.code, payload
    except urllib.error.URLError as exc:
        raise AuthError(f"cannot reach Keycloak: {exc.reason}") from exc


def _decode_claims(access_token: str) -> dict:
    """Read a JWT's payload. No signature check: the token came straight from Keycloak
    over our own server-to-server call, and the cookie we mint from it is what gets
    verified later."""
    try:
        payload = access_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception as exc:
        raise AuthError("malformed token from Keycloak") from exc


def _roles_from_claims(claims: dict) -> list[str]:
    realm_roles = (claims.get("realm_access") or {}).get("roles") or []
    return [r for r in realm_roles if r in _VALID_ROLES]


# ---------------------------------------------------------------- login
def login(username: str, password: str) -> dict:
    """Verify credentials with Keycloak; return a user dict {username, name, roles}."""
    if not enabled():
        raise AuthError("authentication is not configured")
    status, data = _http(
        f"{_realm_base()}/protocol/openid-connect/token", method="POST", form=True,
        data={"grant_type": "password", "client_id": CONFIG.keycloak_client_id,
              "client_secret": CONFIG.keycloak_client_secret,
              "username": username, "password": password, "scope": "openid"})
    if status != 200:
        # 401 = bad creds; anything else is a config/availability problem
        raise AuthError("invalid username or password" if status == 401
                        else f"login failed ({data.get('error_description') or status})")
    claims = _decode_claims(data["access_token"])
    roles = _roles_from_claims(claims)
    if not roles:
        raise AuthError("this account has no PhotoRAW role assigned")
    return {"username": claims.get("preferred_username", username),
            "name": claims.get("name") or claims.get("preferred_username", username),
            "roles": roles}


# ---------------------------------------------------------------- session cookie
def make_session(user: dict) -> str:
    """Sign a stateless session token for a logged-in user dict."""
    payload = {"u": user["username"], "n": user.get("name", user["username"]),
               "r": user["roles"], "exp": int(time.time()) + CONFIG.session_ttl_s}
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig = hmac.new(_SESSION_SECRET, raw.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{raw}.{sig}"


def read_session(cookie: str | None) -> dict | None:
    """Validate a session cookie -> user dict, or None if missing/invalid/expired."""
    if not cookie or "." not in cookie:
        return None
    raw, _, sig = cookie.rpartition(".")
    expect = hmac.new(_SESSION_SECRET, raw.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig, expect):
        return None
    try:
        pad = raw + "=" * (-len(raw) % 4)
        p = json.loads(base64.urlsafe_b64decode(pad))
    except Exception:
        return None
    if p.get("exp", 0) < time.time():
        return None
    return {"username": p["u"], "name": p.get("n", p["u"]), "roles": p.get("r", [])}


def is_super_admin(user: dict | None) -> bool:
    return bool(user and SUPER_ADMIN in user.get("roles", []))


# ---------------------------------------------------------------- Keycloak admin API
_svc_token = {"value": None, "exp": 0.0}


def _service_token() -> str:
    """Client-credentials token for the Admin API (cached until shortly before expiry)."""
    if _svc_token["value"] and _svc_token["exp"] > time.time():
        return _svc_token["value"]
    status, data = _http(
        f"{_realm_base()}/protocol/openid-connect/token", method="POST", form=True,
        data={"grant_type": "client_credentials", "client_id": CONFIG.keycloak_client_id,
              "client_secret": CONFIG.keycloak_client_secret})
    if status != 200:
        raise AuthError("cannot obtain admin token (client needs a service account with "
                        "realm-management roles)")
    _svc_token["value"] = data["access_token"]
    _svc_token["exp"] = time.time() + data.get("expires_in", 60) - 10
    return _svc_token["value"]


def _realm_role(name: str) -> dict:
    status, data = _http(f"{_admin_base()}/roles/{urllib.parse.quote(name)}",
                         token=_service_token())
    if status != 200:
        raise AuthError(f"realm role '{name}' not found in Keycloak")
    return {"id": data["id"], "name": data["name"]}


def list_users() -> list[dict]:
    token = _service_token()
    status, users = _http(f"{_admin_base()}/users?max=500", token=token)
    if status != 200:
        raise AuthError("could not list users")
    out = []
    for u in users:
        st, roles = _http(f"{_admin_base()}/users/{u['id']}/role-mappings/realm", token=token)
        role_names = [r["name"] for r in roles] if st == 200 else []
        out.append({"id": u["id"], "username": u.get("username", ""),
                    "name": " ".join(x for x in (u.get("firstName"), u.get("lastName")) if x)
                             or u.get("username", ""),
                    "email": u.get("email", ""), "enabled": u.get("enabled", True),
                    "roles": [r for r in role_names if r in _VALID_ROLES]})
    out.sort(key=lambda x: x["username"].lower())
    return out


def create_user(username: str, password: str, role: str,
                email: str = "", temporary: bool = False, name: str = "") -> dict:
    if role not in _VALID_ROLES:
        raise AuthError(f"role must be one of {_VALID_ROLES}")
    if len(password) < 6:
        raise AuthError("password must be at least 6 characters")
    if not username:
        raise AuthError("username required")
    token = _service_token()
    # Keycloak's default user profile requires first + last name; populate them (from an
    # optional display name, else the username) so the account is "fully set up" and the
    # user can log in immediately without a Verify-Profile step.
    first, _, last = name.strip().partition(" ")
    # Keycloak's default user profile requires an email; synthesize one when the operator
    # didn't provide it so the account is valid and can log in immediately.
    email = email or f"{username}@photoraw.local"
    status, data = _http(
        f"{_admin_base()}/users", method="POST", token=token,
        data={"username": username, "email": email, "enabled": True,
              "emailVerified": True, "requiredActions": [],   # ready to log in immediately
              "firstName": first or username, "lastName": last or username,
              "credentials": [{"type": "password", "value": password,
                               "temporary": temporary}]})
    if status == 409:
        raise AuthError("a user with that username already exists")
    if status not in (201, 204):
        raise AuthError(f"could not create user ({data.get('errorMessage') or status})")
    # find the new user's id (Keycloak returns it in the Location header, but we
    # relisting by username is simpler and robust)
    st, found = _http(f"{_admin_base()}/users?username={urllib.parse.quote(username)}&exact=true",
                      token=token)
    if st != 200 or not found:
        raise AuthError("user created but could not be looked up to assign its role")
    uid = found[0]["id"]
    role_obj = _realm_role(role)
    st, _ = _http(f"{_admin_base()}/users/{uid}/role-mappings/realm",
                  method="POST", token=token, data=[role_obj])
    if st not in (204, 200):
        raise AuthError("user created but role assignment failed")
    return {"id": uid, "username": username, "roles": [role]}


def set_user_role(user_id: str, role: str) -> None:
    """Make `role` the user's only PhotoRAW role (removes the other one if present)."""
    if role not in _VALID_ROLES:
        raise AuthError(f"role must be one of {_VALID_ROLES}")
    token = _service_token()
    keep = _realm_role(role)
    drop = _realm_role(EDITOR if role == SUPER_ADMIN else SUPER_ADMIN)
    _http(f"{_admin_base()}/users/{user_id}/role-mappings/realm",
          method="POST", token=token, data=[keep])
    _http(f"{_admin_base()}/users/{user_id}/role-mappings/realm",
          method="DELETE", token=token, data=[drop])


def set_user_enabled(user_id: str, enabled_flag: bool) -> None:
    token = _service_token()
    st, _ = _http(f"{_admin_base()}/users/{user_id}", method="PUT", token=token,
                  data={"enabled": enabled_flag})
    if st not in (204, 200):
        raise AuthError("could not update the user")


def delete_user(user_id: str) -> None:
    token = _service_token()
    st, _ = _http(f"{_admin_base()}/users/{user_id}", method="DELETE", token=token)
    if st not in (204, 200):
        raise AuthError("could not delete the user")
