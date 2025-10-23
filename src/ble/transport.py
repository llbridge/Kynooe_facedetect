from __future__ import annotations
import asyncio
import json
import logging
import os
import platform
import time
import threading
import queue
from typing import Optional, Dict, Any, Tuple, List

from bleak import BleakClient, BleakScanner
import config  


def _load_cached_ble(path) -> Tuple[Optional[str], Optional[str]]:
    """Return (address, adapter) from local cache if present."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        addr = (data.get("address") or "").strip().lower() or None
        adp = (data.get("adapter") or "").strip() or None
        return addr, adp
    except Exception:
        return None, None


def _save_cached_ble(path, address: Optional[str], adapter: Optional[str]) -> None:
    """Persist last successful (address, adapter) for fast reconnect."""
    if not address:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"address": address, "adapter": adapter or None, "ts": time.time()}, f)
    except Exception:
        pass


def _resolve_ble_defaults() -> Tuple[Optional[str], Optional[str]]:
    """
    Platform-aware BLE defaults.
    macOS: allow cached/env UUID-like address; scanning still works if missing.
    Linux: prefer env/cached MAC + adapter; fallback to project defaults.
    """
    sys_name = platform.system()
    env_address = os.environ.get("MECHARM_BLE_ADDRESS", "").strip().lower() or None
    env_adapter = os.environ.get("MECHARM_BLE_ADAPTER", "").strip() or None
    cached_addr, cached_adp = _load_cached_ble(config.BLE_CACHE_PATH)
    if sys_name == "Darwin":
        return env_address or cached_addr, env_adapter or cached_adp
    return (
        env_address or cached_addr or config.LINUX_DEFAULT_DEVICE_ADDRESS,
        env_adapter or cached_adp or config.LINUX_DEFAULT_ADAPTER_NAME,
    )


FIXED_DEVICE_ADDRESS, ADAPTER_NAME = _resolve_ble_defaults()


def _compose_role_payload() -> Tuple[bytes, str]:
    """
    Compose the role declaration message.
    Priority:
      1) MECHARM_ROLE_PAYLOAD (verbatim)
      2) MECHARM_ROLE ('master' default) + optional slave list / id
    """
    override = os.environ.get("MECHARM_ROLE_PAYLOAD")
    if override is not None:
        payload = override.strip() or "master"
        return payload.encode("utf-8"), payload

    role = (os.environ.get("MECHARM_ROLE", "master") or "master").strip().lower()
    if role == "master":
        slave_list = (os.environ.get("MECHARM_SLAVE_LIST") or "").strip()
        if not slave_list and config.DEFAULT_SLAVE_ASSIGNMENTS:
            slave_list = ",".join(config.DEFAULT_SLAVE_ASSIGNMENTS)
        message = f"{role}:{slave_list}" if slave_list else role
    elif role == "slave":
        slave_id = (os.environ.get("MECHARM_SLAVE_ID") or "").strip()
        message = f"{role}:{slave_id}" if slave_id else role
    else:
        logging.warning(f"unknown MECHARM_ROLE '{role}', defaulting to 'master'.")
        message = "master"
    return message.encode("utf-8"), message


class BLETransport:
    """Persistent BLE transport with auto-reconnect, throttled writes, and fast reconnect cache."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[BleakClient] = None
        self._device_address: Optional[str] = FIXED_DEVICE_ADDRESS
        self._stop = False
        self._ready = False
        self._connected = False
        self._last_write_ts = 0.0
        self._pending_payload: Optional[Dict[str, Any]] = None
        self._payload_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

    # ---------- public api ----------
    def start(self):
        """Start the background asyncio loop."""
        if self._thread:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run_loop_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """Request stop and disconnect."""
        self._stop = True
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)

    def send_json(self, payload: Dict[str, Any]):
        """Enqueue a JSON-serializable payload to be sent to the joystick characteristic."""
        self._payload_queue.put(payload)

    def connected(self) -> bool:
        """Return True if the link is currently active and ready."""
        return bool(self._connected and self._client and self._client.is_connected)

    # ---------- internal ----------
    def _run_loop_forever(self):
        """Create and run an event loop dedicated to BLE I/O."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._main_task())
        try:
            self._loop.run_forever()
        finally:
            for t in asyncio.all_tasks(loop=self._loop):
                t.cancel()
            self._loop.run_until_complete(asyncio.sleep(0))
            self._loop.close()

    async def _shutdown(self):
        """Gracefully disconnect and stop the loop."""
        self._stop = True
        if self._client and self._client.is_connected:
            try:
                await self._client.disconnect()
            except Exception:
                pass
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    async def _main_task(self):
        """Main state machine: connect, send, and keep the link alive."""
        while not self._stop:
            try:
                if not self._client or not self._client.is_connected:
                    await self._connect_once()
                else:
                    await self._maybe_send_latest()
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.warning(f"ble main loop error: {e}")
                await asyncio.sleep(1.0)

    def _on_disconnected(self, _client):
        """Callback when disconnected by remote or error."""
        self._connected = self._ready = False
        self._pending_payload = None
        while not self._payload_queue.empty():
            try:
                self._payload_queue.get_nowait()
            except queue.Empty:
                break
        logging.warning("ble disconnected.")

    async def _connect_once(self):
        """Attempt fast reconnect first, then fallback to scan if needed."""
        adapter_desc = ADAPTER_NAME or "system-default"
        addr = (self._device_address or "").strip().lower() or None

        async def try_connect(target_addr: str, desc: str) -> bool:
            for addr_type in ("public", "random"):
                try:
                    client_kwargs: Dict[str, Any] = {
                        "timeout": 15.0,
                        "address_type": addr_type,
                        "disconnected_callback": self._on_disconnected,
                    }
                    if ADAPTER_NAME:
                        client_kwargs["adapter"] = ADAPTER_NAME
                    client = BleakClient(target_addr, **client_kwargs)
                    logging.info(f"connecting to {target_addr} ({addr_type}) via {desc}")
                    await client.connect()
                    self._client, self._connected = client, True
                    await asyncio.sleep(0.6)
                    await self._handshake_after_connect()
                    _save_cached_ble(config.BLE_CACHE_PATH, target_addr, ADAPTER_NAME)
                    return True
                except Exception:
                    logging.info(f"connect {addr_type} failed")
            return False

        # fast path: cached/env address
        if addr and await try_connect(addr, f"cached/{adapter_desc}"):
            return

        # fallback: scan and match
        try:
            devices = await BleakScanner.discover(timeout=6.0, adapter=ADAPTER_NAME or None)
        except Exception as e:
            logging.warning(f"scan failed: {e}")
            await asyncio.sleep(1.0)
            return

        def match_target():
            for d in devices:
                if addr and (d.address or "").lower() == addr:
                    return d
            for d in devices:
                uuids = (getattr(d, "metadata", {}) or {}).get("uuids") or []
                if any((u or "").lower() == config.SERVICE_UUID.lower() for u in uuids):
                    return d
            for d in devices:
                if config.TARGET_NAME_KEYWORD in (d.name or "").lower():
                    return d
            return None

        target = match_target()
        if not target:
            logging.warning("no target found in scan, retry later...")
            await asyncio.sleep(1.0)
            return

        if await try_connect(target.address, f"scan/{adapter_desc}"):
            self._device_address = target.address
        else:
            self._connected = self._ready = False

    async def _handshake_after_connect(self):
        """Subscribe (best-effort) and send role payload after a successful connection."""
        if not self.connected():
            return

        # Start notify (best effort)
        for _ in range(3):
            try:
                await self._client.start_notify(config.JOYSTICK_UUID, lambda *_: None)
                break
            except Exception:
                await asyncio.sleep(0.5)

        # Role payload
        role_payload, _ = _compose_role_payload()
        for _ in range(2):
            try:
                await self._client.write_gatt_char(config.ROLE_UUID, role_payload, response=True)
                break
            except Exception:
                await asyncio.sleep(0.4)

        self._ready = True

    async def _maybe_send_latest(self):
        """Send latest queued payload with rate limiting."""
        if not (self.connected() and self._ready):
            return

        if self._pending_payload is None:
            try:
                self._pending_payload = self._payload_queue.get_nowait()
            except queue.Empty:
                return

        now = time.time()
        if (now - self._last_write_ts) < config.MIN_WRITE_INTERVAL_SEC:
            return

        p = self._pending_payload
        payload = {
            "mode": "rectJoystick",
            "x": float(p.get("x", 0.0)),
            "y": float(p.get("y", 0.0)),
            "z": float(p.get("z", 0.0)),
            "delay": int(p.get("delay", 20)),
            "gripper": int(p.get("gripper", 1)),
        }

        try:
            await self._client.write_gatt_char(
                config.JOYSTICK_UUID, json.dumps(payload).encode("utf-8"), response=True
            )
            self._last_write_ts = now
            self._pending_payload = None
        except Exception:
            logging.warning("write failed")
            self._connected = False


def wait_for_ble_connection(
    transport: BLETransport,
    timeout: float = 35.0,
    poll_interval: float = 0.2,
) -> bool:
    """Block until BLE link is connected or timeout elapses."""
    poll_interval = max(0.1, float(poll_interval))
    deadline = time.time() + timeout if (timeout and timeout > 0) else None
    while True:
        if transport.connected():
            return True
        if deadline and time.time() >= deadline:
            return False
        time.sleep(poll_interval)
