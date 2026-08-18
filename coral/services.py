from pathlib import Path
from typing import Optional
from collections import deque
import os, asyncio, time, logging
import yaml

logger = logging.getLogger(__name__)

SERVICES = Path('/workspace/services')
CRASH_LOOP_COUNT = 3
CRASH_LOOP_WINDOW = 60 # seconds

# if a service crashes {CRASH_LOOP_COUNT} times within {CRASH_LOOP_WINDOW} seconds, it is deemed as buggy and does not deserve any more restarts


class Service:
    def __init__(self, manifest: dict, path: Path):

        self.path: Path = path
        self.name: str = manifest['name']
        self.command: list[str] = manifest['command']
        self.cwd: str = manifest.get('cwd') or str(SERVICES)
        self.env: dict = manifest.get('env') or {}
        self.enabled: bool = manifest.get('enabled', True)
        self.autorestart: bool = manifest.get('autorestart', True)

        self.process: Optional[asyncio.subprocess.Process] = None
        self.log_path: Path = SERVICES / f"{self.name}.log"
        self.lifetimes: deque = deque(maxlen=CRASH_LOOP_COUNT)
        self.crash_looped: bool = False
        self._started: Optional[float] = None

    async def start(self):
        SERVICES.mkdir(parents=True, exist_ok=True)
        env = { **os.environ, **{ str(k): str(v) for k, v in self.env.items() } }
        self._started = time.time()
        self.crash_looped = False

        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command, cwd=self.cwd or str(SERVICES), env=env,
                stdout=open(self.log_path, 'ab'), stderr=asyncio.subprocess.STDOUT)

            logger.info("Service %s started (pid %s)", self.name, self.process.pid)

        except Exception:
            logger.exception("Failed to start service %s", self.name)
            self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def stop(self):
        if self.is_running():
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()
            except ProcessLookupError:
                pass

    def status(self) -> dict:
        tail = ''

        if self.log_path.exists():
            try: tail = self.log_path.read_text(errors='replace')[-1500:]
            except Exception: pass

        return {
            'name': self.name, 'command': self.command, 'enabled': self.enabled,
            'running': self.is_running(),
            'pid': self.process.pid if self.is_running() else None,
            'returncode': self.process.returncode if self.process else None,
            'crash_looped': self.crash_looped,
            'uptime_s': round(time.time() - self._started, 1) if (self._started and self.is_running()) else None,
            'log_tail': tail,
        }


class ServiceSupervisor:
    def __init__(self, bot):
        self.bot = bot
        self.services: dict[str, Service] = {}
        self._watchdog_task: Optional[asyncio.Task] = None

    def _manifest_path(self, name: str) -> Path:
        return SERVICES / f"{name}.yaml"

    def load_manifests(self):
        SERVICES.mkdir(parents=True, exist_ok=True)
        self.services.clear()
        for f in sorted(list(SERVICES.glob('*.yaml')) + list(SERVICES.glob('*.yml')) + list(SERVICES.glob('*.json'))):
            try:
                data = yaml.safe_load(f.read_text())  # yaml parses JSON too
                if not data or 'name' not in data or 'command' not in data:
                    logger.warning("Skipping invalid service manifest %s", f); continue
                svc = Service(data, f)
                self.services[svc.name] = svc
            except Exception:
                logger.exception("Failed to load service manifest %s", f)

    async def start_all(self):
        self.load_manifests()
        for svc in self.services.values():
            if svc.enabled:
                await svc.start()
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog())

    async def _watchdog(self):
        while True:
            await asyncio.sleep(5)
            for svc in list(self.services.values()):
                if not svc.enabled or svc.crash_looped or svc.process is None:
                    continue

                if not svc.is_running():
                    lifetime = time.time() - (svc._started or time.time())
                    svc.lifetimes.append(lifetime)

                    if len(svc.lifetimes) == CRASH_LOOP_COUNT and all(t < CRASH_LOOP_WINDOW for t in svc.lifetimes):
                        svc.crash_looped = True
                        logger.warning("Service %s crash-looping; not restarting until reboot or restart_service.", svc.name)
                        continue

                    if svc.autorestart:
                        logger.info("Service %s died (%.1fs); restarting.", svc.name, lifetime)
                        await svc.start()

    def _write_manifest(self, path: Path, data: dict):
        SERVICES.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    async def create(self, name: str, command: list[str], cwd: Optional[str] = None, env: Optional[dict[str, str]] = None, autorestart: bool = True) -> str:

        manifest = {'name': name, 'command': command, 'cwd': cwd or str(SERVICES), 'env': env or {}, 'enabled': True, 'autorestart': autorestart}
        self._write_manifest(self._manifest_path(name), manifest)

        if name in self.services:
            await self.services[name].stop()

        svc = Service(manifest, self._manifest_path(name))
        self.services[name] = svc
        await svc.start()
        return f"Service '{name}' created and started."

    def _patch_enabled(self, svc: Service, enabled: bool):
        try:
            data = yaml.safe_load(svc.path.read_text()); data['enabled'] = enabled
            self._write_manifest(svc.path, data)
        except Exception:
            logger.exception("Failed to update manifest for %s", svc.name)

    async def stop_service(self, name) -> str:
        svc = self.services.get(name)
        if not svc: return f"No service named '{name}'."

        svc.enabled = False; self._patch_enabled(svc, False)
        await svc.stop()
        return f"Service '{name}' stopped."

    async def restart_service(self, name) -> str:
        svc = self.services.get(name)
        if not svc: return f"No service named '{name}'."

        svc.enabled = True; svc.crash_looped = False; svc.lifetimes.clear()
        self._patch_enabled(svc, True)
        await svc.stop(); await svc.start()
        return f"Service '{name}' restarted."

    async def remove(self, name) -> str:
        svc = self.services.pop(name, None)
        if not svc: return f"No service named '{name}'."
        
        await svc.stop()
        try: svc.path.unlink()
        except Exception: pass
        return f"Service '{name}' removed."

    def list_all(self) -> list[dict]:
        return [s.status() for s in self.services.values()]

    def status(self, name) -> dict:
        svc = self.services.get(name)
        return svc.status() if svc else {'error': f"No service named '{name}'."}