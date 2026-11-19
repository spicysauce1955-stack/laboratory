"""A DigitalOcean account shaped like pydo, for teardown tests that must never reach DO.

Modelled on the installed SDK (pydo 0.37 over azure-core) and on SkyPilot 0.13's DO provisioner,
not on our own code, so the tests can catch us misreading either:

* ``volumes.get`` / ``volumes.delete`` / ``droplets.get`` list 404 as an *expected* status: they
  do **not** raise for a missing resource, they return DO's error body
  ``{"id": "not_found", "message": ...}``. A check that waits for an exception never sees "gone".
* every other non-2xx raises ``azure.core.exceptions.HttpResponseError`` (a 409 becomes its
  ``ResourceExistsError`` subclass). Its ``str()`` is ``"(None) <message>\\nCode: None\\n..."``
  and the structured message is on ``.error.message`` — the shape of the 2026-11-19 failure line.
* SkyPilot's ``sky.down`` on DO calls ``droplets.destroy_with_associated_resources_dangerous``:
  DO destroys the droplet and deletes its block volume **asynchronously**, and SkyPilot returns as
  soon as the droplet has left the droplet listing. It never looks at the volume.

Time is a fake clock advanced only by :meth:`FakeDo.sleep`, which tests install over the module's
``time.sleep``; nothing here really waits.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    map_error,
)

# DO's wording on 2026-11-19 (job F), and the wording on 2026-08-23. Both verbatim.
NEW_DETACH = "failed to delete volume: cannot delete attached volume"
OLD_DETACH = "failed to delete volume: attached volume cannot be deleted"
NOT_FOUND = {"id": "not_found", "message": "The resource you were accessing could not be found."}


class _Response:
    """Just enough of an azure-core HTTP response for ``HttpResponseError`` to parse DO's body."""

    def __init__(self, status: int, body: dict[str, Any], reason: str) -> None:
        self.status_code = status
        self.reason = reason
        self.headers: dict[str, str] = {}
        self.content_type = "application/json"
        self._body = json.dumps(body)

    def text(self, encoding: str | None = None) -> str:
        return self._body

    def body(self) -> bytes:
        return self._body.encode()


def http_error(status: int, message: str, *, error_id: str = "unprocessable_entity") -> Exception:
    """The exception pydo raises for ``status``, built by azure-core itself from DO's body."""
    body = {"id": error_id, "message": message, "request_id": "req-test"}
    reason = {409: "Conflict", 422: "Unprocessable Entity", 403: "Forbidden"}.get(status, "Error")
    response = _Response(status, body, reason)
    try:
        map_error(
            status_code=status,
            response=response,  # type: ignore[arg-type]
            error_map={404: ResourceNotFoundError, 409: ResourceExistsError},
        )
        raise HttpResponseError(response=response)  # type: ignore[arg-type]
    except HttpResponseError as e:
        return e


class FakeDo:
    """One DO account: droplets, volumes, an asynchronous detach, and a lagging listing.

    ``detach_s`` is how long after its droplet is destroyed a volume stays attached (``None``:
    forever). ``hide_s`` is how long after a *dangerous* destroy the volume is left out of
    ``volumes.list`` while still existing — the eventual consistency seen live on 2026-11-19.
    ``associated_delete`` is whether DO's background deletion of that volume succeeds.
    """

    def __init__(
        self,
        *,
        volumes: list[dict[str, Any]] = (),  # type: ignore[assignment]
        droplets: list[dict[str, Any]] = (),  # type: ignore[assignment]
        detach_s: float | None = 0.0,
        detach_message: str = NEW_DETACH,
        detach_status: int = 422,
        hide_s: float = 0.0,
        associated_delete: bool = True,
    ) -> None:
        self.now = 0.0
        self.slept: list[float] = []
        self.vols: dict[str, dict[str, Any]] = {str(v["id"]): dict(v) for v in volumes}
        self.live_droplets: dict[int, dict[str, Any]] = {int(d["id"]): dict(d) for d in droplets}
        self.destroyed_at: dict[int, float] = {}
        self.detach_s = detach_s
        self.detach_message = detach_message
        self.detach_status = detach_status
        self.hide_s = hide_s
        self.associated_delete = associated_delete
        self.hidden_until: dict[str, float] = {}
        self.pending_associated: dict[str, float] = {}  # volume id -> when DO deletes it itself
        self.refuse: dict[str, Exception] = {}  # a terminal (non-detach) delete refusal, by id
        self.list_error: Exception | None = None
        self.get_error: Exception | None = None
        self.deleted: list[str] = []
        self.calls: list[str] = []
        self.volumes = SimpleNamespace(list=self._vol_list, get=self._vol_get, delete=self._vol_delete)
        self.droplets = SimpleNamespace(
            list=self._drop_list,
            get=self._drop_get,
            destroy=self._drop_destroy,
            destroy_with_associated_resources_dangerous=self._drop_destroy_dangerous,
        )

    # -- clock ------------------------------------------------------------------------------
    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        self._settle()

    def _settle(self) -> None:
        for vid, when in list(self.pending_associated.items()):
            if self.now >= when and vid in self.vols:
                if self.associated_delete:
                    del self.vols[vid]
                self.pending_associated.pop(vid)

    # -- volumes ----------------------------------------------------------------------------
    def attached(self, vid: str) -> bool:
        vol = self.vols[vid]
        for d in vol.get("droplet_ids") or []:
            if int(d) in self.live_droplets:
                return True
            gone_at = self.destroyed_at.get(int(d))
            if gone_at is not None and (self.detach_s is None or self.now < gone_at + self.detach_s):
                return True
        return False

    def _view(self, vid: str) -> dict[str, Any]:
        vol = dict(self.vols[vid])
        vol["droplet_ids"] = list(vol.get("droplet_ids") or []) if self.attached(vid) else []
        return vol

    def _vol_list(self, per_page: int = 20, **kw: Any) -> dict[str, Any]:
        self.calls.append("volumes.list")
        self._settle()
        if self.list_error is not None:
            raise self.list_error
        shown = [
            self._view(vid) for vid in self.vols if self.now >= self.hidden_until.get(vid, -1.0)
        ]
        return {"volumes": shown, "links": {}, "meta": {"total": len(shown)}}

    def _vol_get(self, volume_id: str, **kw: Any) -> dict[str, Any]:
        self.calls.append(f"volumes.get:{volume_id}")
        self._settle()
        if self.get_error is not None:
            raise self.get_error
        vid = str(volume_id)
        if vid not in self.vols:
            return dict(NOT_FOUND)
        return {"volume": self._view(vid)}

    def _vol_delete(self, volume_id: str, **kw: Any) -> dict[str, Any] | None:
        self.calls.append(f"volumes.delete:{volume_id}")
        self._settle()
        vid = str(volume_id)
        if vid not in self.vols:
            return dict(NOT_FOUND)
        if vid in self.refuse:
            raise self.refuse[vid]
        if self.attached(vid):
            raise http_error(self.detach_status, self.detach_message)
        del self.vols[vid]
        self.pending_associated.pop(vid, None)
        self.deleted.append(vid)
        return None

    # -- droplets ---------------------------------------------------------------------------
    def _drop_list(self, per_page: int = 20, **kw: Any) -> dict[str, Any]:
        self.calls.append("droplets.list")
        return {"droplets": list(self.live_droplets.values()), "links": {}}

    def _drop_get(self, droplet_id: int, **kw: Any) -> dict[str, Any]:
        self.calls.append(f"droplets.get:{droplet_id}")
        d = self.live_droplets.get(int(droplet_id))
        return {"droplet": dict(d)} if d is not None else dict(NOT_FOUND)

    def _drop_destroy(self, droplet_id: int, **kw: Any) -> None:
        """Plain destroy (our DO-direct fallback): the volume detaches but is NOT deleted."""
        self.calls.append(f"droplets.destroy:{droplet_id}")
        self.live_droplets.pop(int(droplet_id), None)
        self.destroyed_at[int(droplet_id)] = self.now

    def _drop_destroy_dangerous(self, droplet_id: int, x_dangerous: bool = False, **kw: Any) -> None:
        """What SkyPilot's ``down_instance`` calls: droplet now, its volumes in the background."""
        self.calls.append(f"droplets.destroy_dangerous:{droplet_id}")
        self._drop_destroy(droplet_id)
        for vid, vol in self.vols.items():
            if int(droplet_id) in [int(d) for d in vol.get("droplet_ids") or []]:
                self.hidden_until[vid] = self.now + self.hide_s
                # DO deletes it once it has detached — or tries to, per `associated_delete`.
                delay = self.detach_s if self.detach_s is not None else float("inf")
                self.pending_associated[vid] = self.now + delay + 0.001


class FakeSky:
    """``sky`` as robust_teardown uses it, where ``down`` does what SkyPilot's DO provisioner does:
    a dangerous destroy of every droplet named after the cluster, returning once they are gone."""

    def __init__(self, do: FakeDo, *, down_error: Exception | None = None) -> None:
        self.do = do
        self.down_error = down_error
        self.downs = 0

    def get(self, x: Any) -> Any:
        return x

    def down(self, cluster: str) -> None:
        self.downs += 1
        if self.down_error is not None:
            raise self.down_error
        for d in list(self.do.live_droplets.values()):
            if str(d["name"]).startswith(cluster):
                self.do.droplets.destroy_with_associated_resources_dangerous(
                    droplet_id=d["id"], x_dangerous=True
                )
